import unittest
import torch
import torch.nn as nn
from adaptive_hybrid_retrieval import GeM, AdaptiveHybridModel, QAFF, AdaptiveHybridRetrieval

class TestAdaptiveHybridRetrieval(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_image = torch.randn(1, 3, 224, 224).to(self.device) # Batch size 1, 3 channels, 224x224
        self.batch_images = torch.randn(4, 3, 224, 224).to(self.device) # Batch size 4
        self.feature_dim = 512
        self.output_dim = 512

    def test_gem_module(self):
        print("\nTesting GeM module...")
        gem_model = GeM().to(self.device)
        # Simulate spatial features from a backbone (e.g., ResNet output before pooling)
        spatial_features = torch.randn(1, 2048, 7, 7).to(self.device) # Example: ResNet50 output
        output = gem_model(spatial_features)
        self.assertEqual(output.shape, (1, 2048, 1, 1)) # Should output 1x1 spatial dimensions
        print("GeM module test passed.")

    def test_adaptive_hybrid_model(self):
        print("\nTesting AdaptiveHybridModel...")
        # Test with resnet18 backbone        model = AdaptiveHybridModel(backbone='resnet18', pretrained=False, output_dim=self.output_dim).to(self.device)        sc_gem, regional_gem, scale_gem = model(self.input_image)
        
        # Check output dimensions
        self.assertEqual(sc_gem.shape, (1, self.output_dim // 3))
        self.assertEqual(regional_gem.shape, (1, self.output_dim // 3))
        self.assertEqual(scale_gem.shape, (1, self.output_dim - 2 * (self.output_dim // 3)))
        
        # Test with batch input
        sc_gem_batch, regional_gem_batch, scale_gem_batch = model(self.batch_images)
        self.assertEqual(sc_gem_batch.shape, (self.batch_images.shape[0], self.output_dim // 3))
        self.assertEqual(regional_gem_batch.shape, (self.batch_images.shape[0], self.output_dim // 3))
        self.assertEqual(scale_gem_batch.shape, (self.batch_images.shape[0], self.output_dim - 2 * (self.output_dim // 3)))
        print("AdaptiveHybridModel test passed.")

    def test_qaff_module(self):
        print("\nTesting QAFF module...")
        qaff_model = QAFF(feature_dim=self.output_dim // 3, num_feature_types=3).to(self.device)
        
        query_feature = torch.randn(1, self.output_dim // 3).to(self.device)
        gallery_features = [
            torch.randn(5, self.output_dim // 3).to(self.device),
            torch.randn(5, self.output_dim // 3).to(self.device),
            torch.randn(5, self.output_dim // 3).to(self.device)
        ]
        
        fused_feature = qaff_model(query_feature, gallery_features)
        self.assertEqual(fused_feature.shape, (5, self.output_dim // 3))
        print("QAFF module test passed.")

    def test_adaptive_hybrid_retrieval(self):
        print("\nTesting AdaptiveHybridRetrieval system...")
        feature_extractor = AdaptiveHybridModel(backbone='resnet18', pretrained=False, output_dim=self.output_dim).to(self.device)
        qaff_module = QAFF(feature_dim=self.output_dim // 3, num_feature_types=3).to(self.device)
        retrieval_system = AdaptiveHybridRetrieval(feature_extractor, qaff_module).to(self.device)

        # Mock gallery data
        gallery_images = torch.randn(10, 3, 224, 224).to(self.device)
        gallery_labels = torch.randint(0, 5, (10,)).to(self.device)
        gallery_paths = [f"path/to/gallery_img_{i}.jpg" for i in range(10)]

        # Add to gallery
        retrieval_system.add_to_gallery(gallery_images, gallery_labels, gallery_paths)
        self.assertIsNotNone(retrieval_system.gallery_sc_gem_embeddings)
        self.assertIsNotNone(retrieval_system.gallery_regional_gem_embeddings)
        self.assertIsNotNone(retrieval_system.gallery_scale_gem_embeddings)
        self.assertEqual(retrieval_system.gallery_sc_gem_embeddings.shape[0], 10)

        # Perform search
        query_image = torch.randn(1, 3, 224, 224).to(self.device)
        scores, indices, retrieved_paths = retrieval_system.search(query_image, top_k=3)
        
        self.assertEqual(scores.shape, (1, 3))
        self.assertEqual(indices.shape, (1, 3))
        self.assertEqual(len(retrieved_paths), 3)
        print("AdaptiveHybridRetrieval system test passed.")

if __name__ == '__main__':
    unittest.main()


