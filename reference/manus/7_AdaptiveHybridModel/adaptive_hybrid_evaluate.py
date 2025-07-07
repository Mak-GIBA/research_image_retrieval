import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
import json
import csv

# 提案手法のモデルをインポート
from adaptive_hybrid_retrieval import AdaptiveHybridModel, QAFF, AdaptiveHybridRetrieval

# 評価用データセットクラス (iris_evaluate.pyから流用)
class RetrievalDataset:
    def __init__(self, name, root_dir=\'./data\'):
        self.name = name
        self.root_dir = os.path.join(root_dir, name)
        self.query_dir = os.path.join(self.root_dir, \'queries\')
        self.gallery_dir = os.path.join(self.root_dir, \'gallery\')
        self.gt_path = os.path.join(self.root_dir, \'ground_truth.json\')
        
        self._create_mock_data()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.query_images, self.query_labels, self.query_paths = self._load_data(self.query_dir)
        self.gallery_images, self.gallery_labels, self.gallery_paths = self._load_data(self.gallery_dir)
        self.ground_truth = self._load_ground_truth()
    
    def _create_mock_data(self):
        os.makedirs(self.query_dir, exist_ok=True)
        os.makedirs(self.gallery_dir, exist_ok=True)
        num_classes = 10
        queries_per_class = 5
        gallery_per_class = 20
        for cls in range(num_classes):
            for i in range(queries_per_class):
                img_path = os.path.join(self.query_dir, f\'query_cls{cls}_img{i}.jpg\')
                if not os.path.exists(img_path):
                    Image.new(\'RGB\', (224, 224), color=(cls*25, 100+i*10, 150)).save(img_path)
            for i in range(gallery_per_class):
                img_path = os.path.join(self.gallery_dir, f\'gallery_cls{cls}_img{i}.jpg\')
                if not os.path.exists(img_path):
                    Image.new(\'RGB\', (224, 224), color=(cls*25, 100+i*5, 150)).save(img_path)
        if not os.path.exists(self.gt_path):
            gt = {}
            for cls in range(num_classes):
                for i in range(queries_per_class):
                    query_id = f\'query_cls{cls}_img{i}\'
                    gt[query_id] = {
                        \'easy\': [f\'gallery_cls{cls}_img{j}\' for j in range(5)],
                        \'medium\': [f\'gallery_cls{cls}_img{j}\' for j in range(5, 10)],
                        \'hard\': [f\'gallery_cls{cls}_img{j}\' for j in range(10, 15)]
                    }
            with open(self.gt_path, \'w\') as f:
                json.dump(gt, f)

    def _load_data(self, data_dir):
        images, labels, paths = [], [], []
        files = sorted(os.listdir(data_dir))
        for file in files:
            if file.endswith(('.jpg', '.png')):
                path = os.path.join(data_dir, file)
                img = Image.open(path).convert(\'RGB\')
                img_tensor = self.transform(img)
                cls = int(file.split(\'_cls\')[1].split(\'_\')[0])
                images.append(img_tensor)
                labels.append(cls)
                paths.append(path)
        return torch.stack(images), torch.tensor(labels), paths

    def _load_ground_truth(self):
        with open(self.gt_path, \'r\') as f:
            return json.load(f)

    def get_query_by_id(self, query_id):
        for i, path in enumerate(self.query_paths):
            if query_id in path:
                return self.query_images[i].unsqueeze(0), self.query_labels[i]
        return None, None

    def get_gallery_indices_by_ids(self, gallery_ids):
        indices = []
        for gallery_id in gallery_ids:
            for i, path in enumerate(self.gallery_paths):
                if gallery_id in path:
                    indices.append(i)
                    break
        return torch.tensor(indices)

# 評価関数 (iris_evaluate.pyから流用・修正)
def evaluate_retrieval_model(retrieval_system, dataset, output_dir):
    device = torch.device(\'cuda\' if torch.cuda.is_available() else \'cpu\')
    retrieval_system.model.to(device)
    retrieval_system.qaff_module.to(device)

    # ギャラリー追加
    retrieval_system.add_to_gallery(dataset.gallery_images.to(device), dataset.gallery_labels, dataset.gallery_paths)

    results = {
        difficulty: {\'mAP\': [], \'P@1\': [], \'P@5\': [], \'P@10\': []} 
        for difficulty in [\'easy\', \'medium\', \'hard\', \'overall\']
    }
    query_results_log = []

    for query_id, gt_dict in tqdm(dataset.ground_truth.items(), desc=f"Evaluating {dataset.name}"):
        query_img, _ = dataset.get_query_by_id(query_id)
        if query_img is None: continue

        scores, indices, paths = retrieval_system.search(query_img.to(device), top_k=100)
        indices = indices.squeeze(0).cpu()

        for difficulty in [\'easy\', \'medium\', \'hard\']:
            gt_indices = dataset.get_gallery_indices_by_ids(gt_dict[difficulty])
            if len(gt_indices) == 0: continue

            ap = compute_ap(indices.tolist(), gt_indices.tolist())
            p_at_k = {k: compute_precision_at_k(indices.tolist(), gt_indices.tolist(), k) for k in [1, 5, 10]}

            results[difficulty][\'mAP\'].append(ap)
            for k in [1, 5, 10]:
                results[difficulty][f\'P@{k}\'].append(p_at_k[k])

            query_results_log.append({
                \'query_id\': query_id, \'difficulty\': difficulty, \'AP\': ap, **p_at_k
            })

    # 平均を計算
    summary = {}
    for difficulty in [\'easy\', \'medium\', \'hard\']:
        summary[difficulty] = {metric: np.mean(values) for metric, values in results[difficulty].items()}
    
    summary[\'overall\'] = {
        metric: np.mean([summary[d][metric] for d in [\'easy\', \'medium\', \'hard\']])
        for metric in [\'mAP\', \'P@1\', \'P@5\', \'P@10\']
    }

    # 結果保存
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f\'{dataset.name}_summary.json\'), \'w\') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(output_dir, f\'{dataset.name}_detailed_results.json\'), \'w\') as f:
        json.dump(query_results_log, f, indent=2)

    visualize_results(summary, dataset.name, output_dir)
    return summary

def compute_ap(ranked_list, gt_list):
    if not gt_list: return 0.0
    hits, sum_prec = 0, 0.0
    for i, p in enumerate(ranked_list):
        if p in gt_list:
            hits += 1
            sum_prec += hits / (i + 1.0)
    return sum_prec / len(gt_list) if gt_list else 0.0

def compute_precision_at_k(ranked_list, gt_list, k):
    if not gt_list: return 0.0
    return len(set(ranked_list[:k]) & set(gt_list)) / k

def visualize_results(summary, dataset_name, output_dir):
    difficulties = [\'easy\', \'medium\', \'hard\', \'overall\']
    map_values = [summary[d][\'mAP\'] for d in difficulties]
    plt.figure(figsize=(10, 6))
    plt.bar(difficulties, map_values, color=[\'green\', \'orange\', \'red\', \'blue\'])
    plt.title(f\'mAP by Difficulty - {dataset_name}\')
    plt.ylabel(\'mAP\'); plt.ylim(0, 1.0)
    plt.grid(axis=\'y\', linestyle=\'--\', alpha=0.7)
    plt.savefig(os.path.join(output_dir, f\'{dataset_name}_map_by_difficulty.png\'))

def main():
    parser = argparse.ArgumentParser(description=\'Evaluate Adaptive Hybrid Retrieval model\')
    parser.add_argument(\'--model_path\', type=str, help=\'Path to the trained model checkpoint\')
    parser.add_argument(\'--datasets\', nargs=\'+\', default=[\'roxford5k\', \'rparis6k\'], help=\'Datasets to evaluate on\')
    parser.add_argument(\'--output_dir\', type=str, default=\'./results_adaptive\', help=\'Directory to save results\')
    parser.add_argument(\'--backbone\', type=str, default=\'resnet18\', help=\'Backbone network\')
    parser.add_argument(\'--dim\', type=int, default=512, help=\'Feature dimension\')
    args = parser.parse_args()

    device = torch.device(\'cuda\' if torch.cuda.is_available() else \'cpu\')
    print(f"Using device: {device}")

    # モデル作成
    feature_extractor = AdaptiveHybridModel(backbone=args.backbone, pretrained=False, output_dim=args.dim)
    qaff_module = QAFF(feature_dim=args.dim // 3) # 各特徴がdim/3の次元を持つと仮定
    
    # 学習済み重みの読み込み (オプション)
    if args.model_path:
        checkpoint = torch.load(args.model_path, map_location=device)
        feature_extractor.load_state_dict(checkpoint[\'feature_extractor_state_dict\'])
        qaff_module.load_state_dict(checkpoint[\'qaff_module_state_dict\'])
        print(f"Loaded model from {args.model_path}")
    else:
        print("No model path provided, using randomly initialized model.")

    retrieval_system = AdaptiveHybridRetrieval(feature_extractor, qaff_module)

    all_results = {}
    for dataset_name in args.datasets:
        print(f"\nEvaluating on {dataset_name}...")
        dataset = RetrievalDataset(dataset_name)
        results = evaluate_retrieval_model(retrieval_system, dataset, args.output_dir)
        all_results[dataset_name] = results
        print(f"\n{dataset_name} Results:")
        for difficulty, metrics in results.items():
            print(f"  {difficulty.capitalize()}: mAP={metrics[\'mAP\']:.4f}, P@1={metrics[\'P@1\']:.4f}, P@5={metrics[\'P@5\']:.4f}")

    with open(os.path.join(args.output_dir, \'all_results.json\'), \'w\') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll evaluation results saved to {args.output_dir}")

if __name__ == \'__main__\':
    main()


