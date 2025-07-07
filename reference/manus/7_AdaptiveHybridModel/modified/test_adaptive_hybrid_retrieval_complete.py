import unittest
import torch
import torch.nn as nn
from adaptive_hybrid_retrieval_complete import GeM, AdaptiveHybridModel, QAFF, AdaptiveHybridRetrieval, ContrastiveLoss

class TestAdaptiveHybridRetrievalComplete(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_image = torch.randn(1, 3, 224, 224).to(self.device)
        self.batch_images = torch.randn(4, 3, 224, 224).to(self.device)
        self.output_dim = 512

    def test_gem_module(self):
        print("\nTesting GeM module...")
        gem_model = GeM().to(self.device)
        spatial_features = torch.randn(1, 2048, 7, 7).to(self.device)
        output = gem_model(spatial_features)
        self.assertEqual(output.shape, (1, 2048, 1, 1))
        print("GeM module test passed.")

    def test_adaptive_hybrid_model(self):
        print("\nTesting AdaptiveHybridModel...")
        model = AdaptiveHybridModel(backbone='resnet18', pretrained=False, output_dim=self.output_dim).to(self.device)
        
        # Test single image
        sc_gem, regional_gem, scale_gem = model(self.input_image)
        
        # Check output dimensions - all should be output_dim
        self.assertEqual(sc_gem.shape, (1, self.output_dim))
        self.assertEqual(regional_gem.shape, (1, self.output_dim))
        self.assertEqual(scale_gem.shape, (1, self.output_dim))
        
        # Test batch input
        sc_gem_batch, regional_gem_batch, scale_gem_batch = model(self.batch_images)
        self.assertEqual(sc_gem_batch.shape, (self.batch_images.shape[0], self.output_dim))
        self.assertEqual(regional_gem_batch.shape, (self.batch_images.shape[0], self.output_dim))
        self.assertEqual(scale_gem_batch.shape, (self.batch_images.shape[0], self.output_dim))
        print("AdaptiveHybridModel test passed.")

    def test_qaff_module(self):
        print("\nTesting QAFF module...")
        qaff_model = QAFF(feature_dim=self.output_dim, num_feature_types=3).to(self.device)
        
        query_feature = torch.randn(1, self.output_dim).to(self.device)
        gallery_features = [
            torch.randn(5, self.output_dim).to(self.device),
            torch.randn(5, self.output_dim).to(self.device),
            torch.randn(5, self.output_dim).to(self.device)
        ]
        
        fused_feature = qaff_model(query_feature, gallery_features)
        self.assertEqual(fused_feature.shape, (5, self.output_dim))
        
        # Test that weights sum to 1
        weights = qaff_model.weight_generator(query_feature)
        self.assertAlmostEqual(weights.sum().item(), 1.0, places=5)
        print("QAFF module test passed.")

    def test_adaptive_hybrid_retrieval(self):
        print("\nTesting AdaptiveHybridRetrieval system...")
        feature_extractor = AdaptiveHybridModel(backbone='resnet18', pretrained=False, output_dim=self.output_dim).to(self.device)
        qaff_module = QAFF(feature_dim=self.output_dim, num_feature_types=3).to(self.device)
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

    def test_contrastive_loss(self):
        print("\nTesting ContrastiveLoss...")
        loss_fn = ContrastiveLoss().to(self.device)
        
        embeddings = torch.randn(8, self.output_dim).to(self.device)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]).to(self.device)
        
        loss = loss_fn(embeddings, labels)
        self.assertIsInstance(loss.item(), float)
        self.assertGreater(loss.item(), 0)
        print("ContrastiveLoss test passed.")

    def test_enhanced_backbone_components(self):
        print("\nTesting enhanced backbone components...")
        from adaptive_hybrid_retrieval_complete import SpatialContextAwareLocalAttention, ChannelwiseDilatedConvolution, EnhancedBackbone
        
        # Test SCALA
        scala = SpatialContextAwareLocalAttention(512).to(self.device)
        x = torch.randn(2, 512, 7, 7).to(self.device)
        out = scala(x)
        self.assertEqual(out.shape, x.shape)
        
        # Test CDConv
        cdconv = ChannelwiseDilatedConvolution(512, 512).to(self.device)
        out = cdconv(x)
        self.assertEqual(out.shape, x.shape)
        
        # Test Enhanced Backbone
        backbone = EnhancedBackbone('resnet18', pretrained=False).to(self.device)
        input_img = torch.randn(2, 3, 224, 224).to(self.device)
        features = backbone(input_img)
        self.assertEqual(features.shape[0], 2)  # Batch size
        self.assertEqual(features.shape[1], 512)  # Feature dimension for ResNet18
        
        print("Enhanced backbone components test passed.")

if __name__ == '__main__':
    unittest.main()

