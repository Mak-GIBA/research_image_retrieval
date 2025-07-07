import numpy as np
import torch
import os
import argparse
from tqdm import tqdm
from collections import defaultdict

# IRIS関連のインポート
from iris_implementation import IRIS, IRISRetrieval

def compute_ap(ranks, nres):
    """
    Average Precision (AP) の計算
    
    引数:
    ranks: 正解画像のランク（0ベース）
    nres: 正解画像の総数
    
    戻り値:
    ap: Average Precision
    """
    num_images = len(ranks)
    ap = 0.0
    recall_step = 1.0 / nres
    
    for index in range(num_images):
        rank = ranks[index]
        if rank == 0:
            precision_0 = 1.0
        else:
            precision_0 = float(index) / rank
        precision_1 = float(index + 1) / (rank + 1)
        ap += (precision_0 + precision_1) * recall_step / 2.0
    
    return ap

def compute_map(ranks, gnd, keeps=None, li=False):
    """
    Mean Average Precision (mAP)およびPrecision@Kを計算する関数

    Parameters
    ----------
    ranks : np.ndarray or list of lists
        検索ランキング結果。
        - li=Falseの場合: shape=(gallery数, query数)の2次元numpy配列。各列が各クエリのランキング（ギャラリー画像インデックスの降順ソート結果）。
        - li=Trueの場合: 各クエリごとのランキングリスト（各要素がギャラリー画像インデックスのリスト）。
        例:
            li=False: ranks[0, 2]はクエリ2に対する1位のギャラリー画像インデックス
            li=True: ranks[2][0]はクエリ2に対する1位のギャラリー画像インデックス

    gnd : list of dict
        クエリごとの正解情報リスト。各要素はdictで、以下のキーを持つ:
            - 'ok': 正解画像インデックス配列（必須）
            - 'junk': 無視画像インデックス配列（任意）
        例: gnd[0]['ok'] = [10, 23, 45]

    keeps : list of int, optional
        Precision@Kを計算するKのリスト（例: [1, 5, 10]）。Noneの場合はmAPのみ計算。

    li : bool, optional
        Trueの場合はranksをリスト形式として扱う（デフォルト: False）。

    Returns
    -------
    mAP : float
        全クエリのMean Average Precision（mAP）

    aps : np.ndarray
        各クエリごとのAverage Precision（shape: [クエリ数]）

    pr : np.ndarray, optional
        keepsが指定された場合、各Kでの平均Precision（shape: [len(keeps)]）

    prs : np.ndarray, optional
        keepsが指定された場合、各クエリ・各KでのPrecision（shape: [クエリ数, len(keeps)]）

    Notes
    -----
    - gndの各要素で'ok'が空の場合、そのクエリは評価対象外（aps/prsは+inf）。
    - junk画像はランキングから除外して評価する。
    - keepsを指定しない場合、mAPとapsのみ返す。

    Examples
    --------
    >>> mAP, aps = compute_map(ranks, gnd)
    >>> mAP, aps, pr, prs = compute_map(ranks, gnd, keeps=[1,5,10])
    >>> mAP, aps, pr, prs = compute_map(ranks_list, gnd, keeps=[1,5,10], li=True)
    """
    if keeps:
        mAP = 0.0
        query_nums = len(gnd)
        aps = np.zeros(query_nums)
        pr = np.zeros(len(keeps))
        prs = np.zeros((query_nums, len(keeps)))
        empty_num = 0
        
        for i in range(query_nums):
            query_gnd_ok = np.array(gnd[i]['ok'])
            
            if query_gnd_ok.shape[0] == 0:
                aps[i] = float('+inf')
                prs[i, :] = float('+inf')
                empty_num += 1
            else:
                try:
                    query_gnd_junk = np.array(gnd[i]['junk'])
                except:
                    query_gnd_junk = np.empty(0)
                
                if li:
                    pos = np.arange(len(ranks[i]))[np.in1d(np.asarray(ranks[i]), query_gnd_ok)]
                    junk = np.arange(len(ranks[i]))[np.in1d(np.asarray(ranks[i]), query_gnd_junk)]
                else:
                    pos = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], query_gnd_ok)]
                    junk = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], query_gnd_junk)]
                
                num = 0
                index = 0
                
                if len(junk):
                    ip = 0
                    while ip < len(pos):
                        while index < len(junk) and pos[ip] > junk[index]:
                            num += 1
                            index += 1
                        pos[ip] -= num
                        ip += 1
                
                # compute ap
                ap = compute_ap(pos, len(query_gnd_ok))
                mAP += ap
                aps[i] = ap
                
                # compute precision @ k
                pos += 1  # convert to 1-based
                for k in range(len(keeps)):
                    kp = min(max(pos), keeps[k])
                    prs[i, k] = (pos <= kp).sum() / kp
                pr += prs[i, :]
        
        mAP = mAP / (query_nums - empty_num)
        pr = pr / (query_nums - empty_num)
        return mAP, aps, pr, prs
    else:
        mAP = 0.0
        query_nums = len(gnd)
        aps = np.zeros(query_nums)
        empty_num = 0
        
        for i in range(query_nums):
            query_gnd_ok = np.array(gnd[i]['ok'])
            
            if query_gnd_ok.shape[0] == 0:
                aps[i] = float('+inf')
                empty_num += 1
            else:
                try:
                    query_gnd_junk = np.array(gnd[i]['junk'])
                except:
                    query_gnd_junk = np.empty(0)
                
                if li:
                    pos = np.arange(len(ranks[i]))[np.in1d(np.asarray(ranks[i]), query_gnd_ok)]
                    junk = np.arange(len(ranks[i]))[np.in1d(np.asarray(ranks[i]), query_gnd_junk)]
                else:
                    pos = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], query_gnd_ok)]
                    junk = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], query_gnd_junk)]
                
                num = 0
                index = 0
                
                if len(junk):
                    ip = 0
                    while ip < len(pos):
                        while index < len(junk) and pos[ip] > junk[index]:
                            num += 1
                            index += 1
                        pos[ip] -= num
                        ip += 1
                
                # compute ap
                ap = compute_ap(pos, len(query_gnd_ok))
                mAP += ap
                aps[i] = ap
        
        mAP = mAP / (query_nums - empty_num)
        return mAP, aps

def compute_map_and_print(dataset, featuretype, mode, ranks, gnd, kappas=[1, 5, 10], verbose=False, li=False):
    """
    mAPを計算し、結果を表示する
    
    引数:
    dataset: データセット名
    featuretype: 特徴量タイプ
    mode: 評価モード
    ranks: ランキング結果
    gnd: 正解データ
    kappas: Precision@K を計算する K のリスト
    verbose: 詳細な結果を表示するかどうか
    li: リスト形式のランキングを使用するかどうか
    
    戻り値:
    mapE, mapM, mapH: Easy, Medium, Hard の mAP
    """
    # 旧評価プロトコル
    if dataset.startswith('oxford5k') or dataset.startswith('paris6k'):
        map, aps, _, _ = compute_map(ranks, gnd)
        print('>> {}: mAP {:.2f}'.format(dataset, np.around(map * 100, decimals=2)))
        return np.around(map * 100, decimals=2), None, None
    
    # 新評価プロトコル
    elif dataset.startswith('roxford5k') or dataset.startswith('rparis6k'):
        # Easy
        gnd_t = []
        for i in range(len(gnd)):
            g = {}
            g['ok'] = np.concatenate([gnd[i]['easy']])
            g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['hard']])
            gnd_t.append(g)
        mapE, apsE, mprE, prsE = compute_map(ranks, gnd_t, kappas, li=li)
        
        # Medium
        gnd_t = []
        for i in range(len(gnd)):
            g = {}
            g['ok'] = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
            g['junk'] = np.concatenate([gnd[i]['junk']])
            gnd_t.append(g)
        mapM, apsM, mprM, prsM = compute_map(ranks, gnd_t, kappas, li=li)
        
        # Hard
        gnd_t = []
        for i in range(len(gnd)):
            g = {}
            g['ok'] = np.concatenate([gnd[i]['hard']])
            g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['easy']])
            gnd_t.append(g)
        mapH, apsH, mprH, prsH = compute_map(ranks, gnd_t, kappas, li=li)
        
        print('>> Test Dataset: {} *** Feature Type: {} >>'.format(dataset, featuretype))
        print('>> mAP Easy: {}, Medium: {}, Hard: {}'.format(
            np.around(mapE * 100, decimals=2),
            np.around(mapM * 100, decimals=2),
            np.around(mapH * 100, decimals=2)
        ))
        print('>> mP@k{} Easy: {}, Medium: {}, Hard: {}'.format(
            kappas,
            np.around(mprE * 100, decimals=2),
            np.around(mprM * 100, decimals=2),
            np.around(mprH * 100, decimals=2)
        ))
        
        if verbose:
            print('>> Query aps: >>\nEasy: {}\nMedium: {}\nHard: {}'.format(
                np.around(apsE * 100, decimals=2),
                np.around(apsM * 100, decimals=2),
                np.around(apsH * 100, decimals=2)
            ))
        
        return np.around(mapE * 100, decimals=2), np.around(mapM * 100, decimals=2), np.around(mapH * 100, decimals=2)
    
    else:
        print(f"Unknown dataset: {dataset}")
        return None, None, None

def extract_features(model, dataloader, device):
    """
    データローダーから特徴量を抽出する
    
    引数:
    model: 特徴抽出モデル
    dataloader: データローダー
    device: デバイス
    
    戻り値:
    features: 抽出された特徴量
    labels: 対応するラベル
    """
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            outputs = model(images)
            features.append(outputs['features'].cpu())
            labels.append(targets)
    
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    
    return features, labels

def load_dataset(dataset_name, transform=None):
    """
    データセットを読み込む
    
    引数:
    dataset_name: データセット名
    transform: 前処理
    
    戻り値:
    query_loader: クエリデータローダー
    gallery_loader: ギャラリーデータローダー
    gnd: 正解データ
    """
    # 実際の実装では、ここで指定されたデータセットを読み込む
    # この例では、モックデータを使用
    
    # モックデータの作成
    if dataset_name == 'roxford5k':
        num_queries = 70
        num_gallery = 5000
    elif dataset_name == 'rparis6k':
        num_queries = 70
        num_gallery = 6000
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # モッククエリとギャラリーの作成
    query_features = torch.randn(num_queries, 512)
    gallery_features = torch.randn(num_gallery, 512)
    
    # モック正解データの作成
    gnd = []
    for i in range(num_queries):
        g = {}
        # 各クエリに対して、ランダムな正解画像を生成
        g['easy'] = np.random.choice(num_gallery, size=5, replace=False)
        g['hard'] = np.random.choice(num_gallery, size=5, replace=False)
        g['junk'] = np.random.choice(num_gallery, size=5, replace=False)
        gnd.append(g)
    
    return query_features, gallery_features, gnd

def evaluate_model(model_path, dataset_name, batch_size=16, num_workers=4, device='cuda'):
    """
    モデルを評価する
    
    引数:
    model_path: モデルのパス
    dataset_name: データセット名
    batch_size: バッチサイズ
    num_workers: ワーカー数
    device: デバイス
    
    戻り値:
    results: 評価結果
    """
    # デバイスの設定
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # モデルの読み込み
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # モデルの初期化
    model = IRIS(
        backbone='resnet18',
        pretrained=False,
        dim=512,
        num_classes=1000,  # 実際のクラス数に合わせて調整
        use_spatial_features=True
    )
    
    # モデルの重みを読み込み
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded. Evaluating on {dataset_name}")
    
    # データセットの読み込み
    query_features, gallery_features, gnd = load_dataset(dataset_name)
    
    # 特徴量の正規化
    query_features = torch.nn.functional.normalize(query_features, p=2, dim=1)
    gallery_features = torch.nn.functional.normalize(gallery_features, p=2, dim=1)
    
    # 類似度の計算
    similarity = torch.mm(query_features, gallery_features.t()).cpu().numpy()
    
    # ランキングの計算
    ranks = np.argsort(-similarity, axis=1)
    
    # mAPの計算
    print(f"Computing mAP for {dataset_name}")
    mapE, mapM, mapH = compute_map_and_print(
        dataset=dataset_name,
        featuretype='IRIS',
        mode='global',
        ranks=ranks,
        gnd=gnd,
        kappas=[1, 5, 10],
        verbose=True
    )
    
    # 結果の保存
    results = {
        'dataset': dataset_name,
        'mapE': mapE,
        'mapM': mapM,
        'mapH': mapH
    }
    
    return results

def main():
    parser = argparse.ArgumentParser(description='IRIS Evaluation Script')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--datasets', nargs='+', default=['roxford5k', 'rparis6k'], help='Datasets to evaluate on')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    
    args = parser.parse_args()
    
    # 出力ディレクトリの作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 全データセットでの評価結果
    all_results = {}
    
    # 各データセットで評価
    for dataset_name in args.datasets:
        print(f"\n=== Evaluating on {dataset_name} ===")
        results = evaluate_model(
            model_path=args.model_path,
            dataset_name=dataset_name,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device
        )
        all_results[dataset_name] = results
        
        # 結果の保存
        output_file = os.path.join(args.output_dir, f'{dataset_name}_results.txt')
        with open(output_file, 'w') as f:
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"mAP Easy: {results['mapE']}\n")
            f.write(f"mAP Medium: {results['mapM']}\n")
            f.write(f"mAP Hard: {results['mapH']}\n")
        
        print(f"Results saved to {output_file}")
    
    # 全体の結果をCSVに保存
    output_csv = os.path.join(args.output_dir, 'all_results.csv')
    with open(output_csv, 'w') as f:
        f.write("Dataset,mAP Easy,mAP Medium,mAP Hard\n")
        for dataset_name, results in all_results.items():
            f.write(f"{dataset_name},{results['mapE']},{results['mapM']},{results['mapH']}\n")
    
    print(f"\nAll results saved to {output_csv}")
    
    # 結果の表示
    print("\n=== Summary of Results ===")
    for dataset_name, results in all_results.items():
        print(f"Dataset: {dataset_name}")
        print(f"  mAP Easy: {results['mapE']}")
        print(f"  mAP Medium: {results['mapM']}")
        print(f"  mAP Hard: {results['mapH']}")

if __name__ == '__main__':
    main()
