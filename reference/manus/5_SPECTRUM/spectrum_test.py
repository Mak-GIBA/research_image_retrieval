import torch
import unittest
from spectrum_implementation import (
    CASTLE, PRISM, NEXUS, ORACLE, HARMONY, SPECTRUM, SPECTRUMLoss,
    compute_similarity, evaluate_retrieval
)

class TestCASTLE(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.dim = 128
        self.castle = CASTLE(dim=self.dim, num_heads=4)
        self.dummy_features = torch.randn(self.batch_size, self.dim)
        
    def test_forward(self):
        """Test CASTLE forward pass"""
        output = self.castle(self.dummy_features)
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.dim))
        # Check output is not None
        self.assertIsNotNone(output)
        # Check output has valid values (no NaN or inf)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
        
    def test_causal_mask(self):
        """Test causal mask computation"""
        mask = self.castle.compute_causal_mask(self.dummy_features)
        # Check mask shape
        self.assertEqual(mask.shape, (self.batch_size, self.batch_size))
        # Check mask values are binary (0 or 1)
        unique_values = torch.unique(mask)
        self.assertTrue(all(val in [0.0, 1.0] for val in unique_values))
        # Check diagonal is 1 (self-causality)
        diag = torch.diag(mask)
        self.assertTrue(torch.all(diag == 1.0))


class TestPRISM(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.visual_dim = 128
        self.text_dim = 64
        self.output_dim = 128
        self.prism = PRISM(
            visual_dim=self.visual_dim,
            text_dim=self.text_dim,
            output_dim=self.output_dim
        )
        self.dummy_visual = torch.randn(self.batch_size, self.visual_dim)
        self.dummy_text = torch.randn(self.batch_size, self.text_dim)
        
    def test_forward_with_text(self):
        """Test PRISM forward pass with provided text features"""
        output = self.prism(self.dummy_visual, self.dummy_text)
        # Check output is a dictionary
        self.assertIsInstance(output, dict)
        # Check features key exists
        self.assertIn('features', output)
        # Check features shape
        self.assertEqual(output['features'].shape, (self.batch_size, self.output_dim))
        # Check semantic_similarity exists and has correct shape
        self.assertIn('semantic_similarity', output)
        self.assertEqual(output['semantic_similarity'].shape, (self.batch_size, self.batch_size))
        
    def test_forward_without_text(self):
        """Test PRISM forward pass without text features (should generate them)"""
        output = self.prism(self.dummy_visual)
        # Check output is a dictionary
        self.assertIsInstance(output, dict)
        # Check features key exists
        self.assertIn('features', output)
        # Check features shape
        self.assertEqual(output['features'].shape, (self.batch_size, self.output_dim))
        
    def test_cross_modal_attention(self):
        """Test cross-modal attention computation"""
        attended_text, attended_vis = self.prism.cross_modal_attention(self.dummy_visual, self.dummy_text)
        # Check output shapes
        self.assertEqual(attended_text.shape, (self.batch_size, self.text_dim))
        self.assertEqual(attended_vis.shape, (self.batch_size, self.text_dim))


class TestNEXUS(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.seq_len = 16  # Sequence length for tokenized features
        self.dim = 128
        self.nexus = NEXUS(
            dim=self.dim,
            min_window_size=2,
            max_window_size=4
        )
        # Create tokenized features [B, N, D]
        self.dummy_tokens = torch.randn(self.batch_size, self.seq_len, self.dim)
        
    def test_forward(self):
        """Test NEXUS forward pass"""
        output = self.nexus(self.dummy_tokens)
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.dim))
        # Check output has valid values
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
        
    def test_neural_sparse_attention_mask(self):
        """Test neural sparse attention mask generation"""
        # Create dummy attention map [B, num_heads, N, N]
        dummy_attn = torch.rand(self.batch_size, 4, self.seq_len, self.seq_len)
        mask = self.nexus.neural_sparse_attention_mask(dummy_attn)
        # Check mask shape
        self.assertEqual(mask.shape, dummy_attn.shape)
        # Check mask values are binary (0 or 1)
        unique_values = torch.unique(mask)
        self.assertTrue(all(val in [0.0, 1.0] for val in unique_values))


class TestORACLE(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.dim = 128
        self.num_objects = 6
        self.relation_dim = 64
        self.oracle = ORACLE(
            dim=self.dim,
            num_objects=self.num_objects,
            relation_dim=self.relation_dim
        )
        self.dummy_features = torch.randn(self.batch_size, self.dim)
        
    def test_forward(self):
        """Test ORACLE forward pass"""
        output = self.oracle(self.dummy_features)
        # Check output is a dictionary
        self.assertIsInstance(output, dict)
        # Check features key exists
        self.assertIn('features', output)
        # Check features shape
        self.assertEqual(output['features'].shape, (self.batch_size, self.dim))
        # Check object_features exists and has correct shape
        self.assertIn('object_features', output)
        self.assertEqual(output['object_features'].shape, (self.batch_size, self.num_objects, self.dim))
        
    def test_extract_object_features(self):
        """Test object feature extraction"""
        obj_features = self.oracle.extract_object_features(self.dummy_features)
        # Check output shape
        self.assertEqual(obj_features.shape, (self.batch_size, self.num_objects, self.dim))
        
    def test_dual_branch_processing(self):
        """Test dual branch processing"""
        obj_features = self.oracle.extract_object_features(self.dummy_features)
        obj_branch, rel_branch = self.oracle.dual_branch_processing(obj_features)
        # Check output shapes
        self.assertEqual(obj_branch.shape, (self.batch_size, self.dim))
        self.assertEqual(rel_branch.shape, (self.batch_size, self.dim))


class TestHARMONY(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.visual_dim = 128
        self.text_dim = 64
        self.output_dim = 128
        self.harmony = HARMONY(
            visual_dim=self.visual_dim,
            text_dim=self.text_dim,
            output_dim=self.output_dim
        )
        self.dummy_visual = torch.randn(self.batch_size, self.visual_dim)
        self.dummy_text = torch.randn(self.batch_size, self.text_dim)
        self.dummy_teacher = torch.randn(self.batch_size, self.output_dim)
        
    def test_forward(self):
        """Test HARMONY forward pass"""
        output = self.harmony(self.dummy_visual, self.dummy_text, self.dummy_teacher)
        # Check output is a dictionary
        self.assertIsInstance(output, dict)
        # Check features key exists
        self.assertIn('features', output)
        # Check features shape
        self.assertEqual(output['features'].shape, (self.batch_size, self.output_dim))
        # Check loss components exist
        self.assertIn('kd_loss', output)
        self.assertIn('harmony_cosine_sim', output)
        self.assertIn('harmony_mse_loss', output)
        
    def test_hierarchical_modality_fusion(self):
        """Test hierarchical modality fusion"""
        fused = self.harmony.hierarchical_modality_fusion(self.dummy_visual, self.dummy_text)
        # Check output shape
        self.assertEqual(fused.shape, (self.batch_size, self.output_dim))
        
    def test_harmonic_optimization(self):
        """Test harmonic optimization loss components"""
        losses = self.harmony.harmonic_optimization(self.dummy_visual, self.dummy_text)
        # Check output is a dictionary
        self.assertIsInstance(losses, dict)
        # Check loss components exist
        self.assertIn('harmony_cosine_sim', losses)
        self.assertIn('harmony_mse_loss', losses)
        self.assertIn('harmony_reg_loss', losses)


class TestSPECTRUM(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.img_channels = 3
        self.img_size = 224
        self.dim = 128
        self.text_dim = 64
        self.num_classes = 100
        
        # Initialize SPECTRUM with all modules
        self.spectrum_all = SPECTRUM(
            backbone='resnet18',
            dim=self.dim,
            text_dim=self.text_dim,
            num_classes=self.num_classes,
            use_castle=True,
            use_prism=True,
            use_nexus=False,  # Disabled as it requires tokenized features
            use_oracle=True,
            use_harmony=True,
            module_selection=True
        )
        
        # Initialize SPECTRUM with minimal modules
        self.spectrum_minimal = SPECTRUM(
            backbone='resnet18',
            dim=self.dim,
            text_dim=self.text_dim,
            num_classes=self.num_classes,
            use_castle=True,
            use_prism=False,
            use_nexus=False,
            use_oracle=False,
            use_harmony=False,
            module_selection=False
        )
        
        # Create dummy inputs
        self.dummy_images = torch.randn(self.batch_size, self.img_channels, self.img_size, self.img_size)
        self.dummy_text = torch.randn(self.batch_size, self.text_dim)
        self.dummy_labels = torch.randint(0, self.num_classes, (self.batch_size,))
        
    def test_forward_all_modules(self):
        """Test SPECTRUM forward pass with all modules"""
        with torch.no_grad():  # No need for gradients in testing
            output = self.spectrum_all(self.dummy_images, text_input=self.dummy_text)
        
        # Check output is a dictionary
        self.assertIsInstance(output, dict)
        # Check required keys exist
        self.assertIn('features', output)
        self.assertIn('raw_features', output)
        self.assertIn('logits', output)
        self.assertIn('module_outputs', output)
        self.assertIn('module_weights', output)
        
        # Check shapes
        self.assertEqual(output['features'].shape, (self.batch_size, self.dim))
        self.assertEqual(output['raw_features'].shape, (self.batch_size, self.dim))
        self.assertEqual(output['logits'].shape, (self.batch_size, self.num_classes))
        
        # Check module outputs
        module_outputs = output['module_outputs']
        self.assertIn('castle', module_outputs)
        self.assertIn('prism', module_outputs)
        self.assertIn('oracle', module_outputs)
        self.assertIn('harmony', module_outputs)
        
        # Check module weights
        module_weights = output['module_weights']
        self.assertIn('castle', module_weights)
        self.assertIn('prism', module_weights)
        self.assertIn('oracle', module_weights)
        self.assertIn('harmony', module_weights)
        
    def test_forward_minimal_modules(self):
        """Test SPECTRUM forward pass with minimal modules"""
        with torch.no_grad():
            output = self.spectrum_minimal(self.dummy_images)
        
        # Check output is a dictionary
        self.assertIsInstance(output, dict)
        # Check required keys exist
        self.assertIn('features', output)
        self.assertIn('raw_features', output)
        self.assertIn('logits', output)
        self.assertIn('module_outputs', output)
        
        # Check shapes
        self.assertEqual(output['features'].shape, (self.batch_size, self.dim))
        self.assertEqual(output['raw_features'].shape, (self.batch_size, self.dim))
        self.assertEqual(output['logits'].shape, (self.batch_size, self.num_classes))
        
        # Check module outputs
        module_outputs = output['module_outputs']
        self.assertIn('castle', module_outputs)
        self.assertNotIn('prism', module_outputs)
        self.assertNotIn('oracle', module_outputs)
        self.assertNotIn('harmony', module_outputs)
        
    def test_get_embedding(self):
        """Test get_embedding method"""
        with torch.no_grad():
            embeddings = self.spectrum_all.get_embedding(self.dummy_images, text_input=self.dummy_text)
        
        # Check output shape
        self.assertEqual(embeddings.shape, (self.batch_size, self.dim))
        # Check embeddings are normalized
        norms = torch.norm(embeddings, p=2, dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-6))


class TestSPECTRUMLoss(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.dim = 128
        self.num_classes = 100
        
        # Initialize loss function
        self.loss_fn = SPECTRUMLoss(
            task_loss_type='cross_entropy',
            task_weight=1.0,
            kd_weight=0.5,
            harmony_cosine_weight=0.3,
            harmony_mse_weight=0.3,
            harmony_reg_weight=0.2
        )
        
        # Create dummy outputs and targets
        self.dummy_outputs = {
            'features': torch.randn(self.batch_size, self.dim),
            'logits': torch.randn(self.batch_size, self.num_classes),
            'kd_loss': torch.tensor(0.5),
            'harmony_cosine_sim': torch.tensor(0.8),
            'harmony_mse_loss': torch.tensor(0.3),
            'harmony_reg_loss': torch.tensor(0.2)
        }
        
        self.dummy_targets = {
            'labels': torch.randint(0, self.num_classes, (self.batch_size,))
        }
        
    def test_loss_calculation(self):
        """Test loss calculation"""
        loss_dict = self.loss_fn(self.dummy_outputs, self.dummy_targets)
        
        # Check output is a dictionary
        self.assertIsInstance(loss_dict, dict)
        # Check required keys exist
        self.assertIn('task_loss', loss_dict)
        self.assertIn('kd_loss', loss_dict)
        self.assertIn('harmony_cos_loss', loss_dict)
        self.assertIn('harmony_mse_loss', loss_dict)
        self.assertIn('harmony_reg_loss', loss_dict)
        self.assertIn('total_loss', loss_dict)
        
        # Check loss values are valid
        for key, value in loss_dict.items():
            self.assertFalse(torch.isnan(value))
            self.assertFalse(torch.isinf(value))
            self.assertTrue(value >= 0.0 or key == 'harmony_cos_loss')  # harmony_cos_loss can be negative


class TestUtilityFunctions(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.gallery_size = 10
        self.dim = 128
        
        # Create dummy embeddings
        self.query_embedding = torch.randn(1, self.dim)
        self.gallery_embeddings = torch.randn(self.gallery_size, self.dim)
        
        # Create dummy labels
        self.query_labels = torch.tensor([0])
        self.gallery_labels = torch.randint(0, 5, (self.gallery_size,))
        # Make sure at least one gallery item matches the query
        self.gallery_labels[0] = 0
        
    def test_compute_similarity(self):
        """Test similarity computation"""
        similarity = compute_similarity(self.query_embedding, self.gallery_embeddings)
        
        # Check output shape
        self.assertEqual(similarity.shape, (1, self.gallery_size))
        # Check similarity values are between -1 and 1
        self.assertTrue(torch.all(similarity >= -1.0))
        self.assertTrue(torch.all(similarity <= 1.0))
        
    def test_evaluate_retrieval(self):
        """Test retrieval evaluation"""
        # Batch of query embeddings
        query_embeddings = torch.
(Content truncated due to size limit. Use line ranges to read in chunks)