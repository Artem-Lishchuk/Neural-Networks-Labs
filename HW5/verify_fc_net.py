"""
Verification script for FullyConnectedNet implementation.
Tests initialization, forward pass, backward pass, and gradient correctness.
"""
import torch
import sys
import traceback

from layers import *
from layer_utils import *
from fc_net import FullyConnectedNet, TwoLayerNet
from gradient_check import eval_numerical_gradient

def rel_error(x, y):
    return ((x - y).abs() / (x.abs() + y.abs() + 1e-8).clamp(min=1e-8)).max().item()

passed = 0
failed = 0
errors = []

def run_test(name, fn):
    global passed, failed, errors
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    try:
        fn()
        passed += 1
        print(f"  >> PASSED")
    except Exception as e:
        failed += 1
        errors.append((name, str(e)))
        print(f"  >> FAILED: {e}")
        traceback.print_exc()

# ============================================================
# TEST 1: Parameter initialization shapes and values
# ============================================================
def test_init_shapes():
    torch.manual_seed(42)
    D, C = 30, 10
    hidden = [50, 40, 30]
    model = FullyConnectedNet(hidden, input_dim=D, num_classes=C,
                              weight_scale=1e-2, dtype=torch.float64)

    expected_shapes = {
        'W1': (30, 50), 'b1': (50,),
        'W2': (50, 40), 'b2': (40,),
        'W3': (40, 30), 'b3': (30,),
        'W4': (30, 10), 'b4': (10,),
    }

    assert model.num_layers == 4, f"num_layers should be 4, got {model.num_layers}"

    for name, shape in expected_shapes.items():
        assert name in model.params, f"Missing parameter: {name}"
        assert model.params[name].shape == shape, \
            f"{name} shape mismatch: expected {shape}, got {model.params[name].shape}"

    for i in range(1, 5):
        b = model.params[f'b{i}']
        assert torch.all(b == 0), f"b{i} should be initialized to zero"

    for i in range(1, 5):
        w = model.params[f'W{i}']
        assert w.dtype == torch.float64, f"W{i} dtype should be float64, got {w.dtype}"
        assert w.std().item() < 0.05, f"W{i} std too large: {w.std().item():.4f}"

    print(f"  num_layers = {model.num_layers}")
    print(f"  Parameters: {sorted(model.params.keys())}")
    for k, v in sorted(model.params.items()):
        print(f"    {k}: shape={tuple(v.shape)}, dtype={v.dtype}, mean={v.mean():.6f}, std={v.std():.6f}")

# ============================================================
# TEST 2: Forward pass produces correct output shape
# ============================================================
def test_forward_shape():
    torch.manual_seed(42)
    N, D, C = 5, 30, 10
    hidden = [50, 40]
    model = FullyConnectedNet(hidden, input_dim=D, num_classes=C,
                              weight_scale=1e-2, dtype=torch.float64)

    X = torch.randn(N, D, dtype=torch.float64)
    scores = model.loss(X)

    assert scores.shape == (N, C), f"Scores shape mismatch: expected ({N}, {C}), got {scores.shape}"
    print(f"  scores shape: {scores.shape}  (correct)")

# ============================================================
# TEST 3: Initial loss sanity check (should be ~-ln(1/C) for random weights)
# ============================================================
def test_initial_loss():
    torch.manual_seed(42)
    N, D, C = 50, 30, 10
    hidden = [50, 40]
    model = FullyConnectedNet(hidden, input_dim=D, num_classes=C,
                              reg=0.0, weight_scale=1e-2, dtype=torch.float64)

    X = torch.randn(N, D, dtype=torch.float64)
    y = torch.randint(C, (N,))

    loss, grads = model.loss(X, y)
    expected_loss = -torch.log(torch.tensor(1.0 / C)).item()

    print(f"  Initial loss: {loss:.4f}")
    print(f"  Expected (approx): {expected_loss:.4f}")
    assert abs(loss - expected_loss) < 1.0, \
        f"Initial loss {loss:.4f} is too far from expected {expected_loss:.4f}"

# ============================================================
# TEST 4: Regularization increases loss
# ============================================================
def test_regularization_increases_loss():
    torch.manual_seed(42)
    N, D, C = 10, 20, 5
    hidden = [30, 20]
    X = torch.randn(N, D, dtype=torch.float64)
    y = torch.randint(C, (N,))

    model_noreg = FullyConnectedNet(hidden, input_dim=D, num_classes=C,
                                    reg=0.0, weight_scale=1e-2, dtype=torch.float64)
    model_reg = FullyConnectedNet(hidden, input_dim=D, num_classes=C,
                                  reg=1.0, weight_scale=1e-2, dtype=torch.float64)

    # Copy weights to make comparison fair
    for k in model_noreg.params:
        model_reg.params[k] = model_noreg.params[k].clone()

    loss_noreg, _ = model_noreg.loss(X, y)
    loss_reg, _ = model_reg.loss(X, y)

    print(f"  Loss without reg: {loss_noreg:.4f}")
    print(f"  Loss with reg=1.0: {loss_reg:.4f}")
    assert loss_reg > loss_noreg, "Regularization should increase the loss"

# ============================================================
# TEST 5: Gradient check with numerical gradients (THE KEY TEST)
# ============================================================
def test_gradient_check():
    torch.manual_seed(231)
    N, D, H1, H2, C = 2, 15, 20, 30, 10
    X = torch.randn(N, D, dtype=torch.float64)
    y = torch.randint(C, (N,))

    for reg in [0, 3.14]:
        print(f"  --- reg={reg} ---")
        model = FullyConnectedNet([H1, H2], input_dim=D, num_classes=C,
                                  reg=reg, weight_scale=5e-2, dtype=torch.float64)

        loss, grads = model.loss(X, y)
        print(f"  loss = {loss:.6f}")

        for name in sorted(grads):
            f = lambda _: model.loss(X, y)[0]
            grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
            error = rel_error(grad_num, grads[name])
            status = "OK" if error < 1e-4 else "FAIL"
            print(f"    {name} relative error: {error:.2e}  [{status}]")
            assert error < 1e-4, f"Gradient check failed for {name}: rel_error={error:.2e}"

# ============================================================
# TEST 6: Single hidden layer should match TwoLayerNet
# ============================================================
def test_matches_two_layer_net():
    torch.manual_seed(42)
    N, D, H, C = 5, 20, 30, 10
    X = torch.randn(N, D, dtype=torch.float32)
    y = torch.randint(C, (N,))

    two_layer = TwoLayerNet(input_dim=D, hidden_dim=H, num_classes=C,
                            weight_scale=1e-2, reg=0.5)
    fc_net = FullyConnectedNet([H], input_dim=D, num_classes=C,
                               reg=0.5, weight_scale=1e-2, dtype=torch.float32)

    # Copy the exact same weights
    fc_net.params['W1'] = two_layer.params['W1'].clone()
    fc_net.params['b1'] = two_layer.params['b1'].clone()
    fc_net.params['W2'] = two_layer.params['W2'].clone()
    fc_net.params['b2'] = two_layer.params['b2'].clone()

    loss_two, grads_two = two_layer.loss(X, y)
    loss_fc, grads_fc = fc_net.loss(X, y)

    print(f"  TwoLayerNet loss:        {loss_two:.6f}")
    print(f"  FullyConnectedNet loss:  {loss_fc:.6f}")
    print(f"  Loss difference:         {abs(loss_two - loss_fc):.2e}")

    assert abs(loss_two.item() - loss_fc.item()) < 1e-10, \
        f"Losses should match: {loss_two:.6f} vs {loss_fc:.6f}"

    for name in ['W1', 'b1', 'W2', 'b2']:
        err = rel_error(grads_two[name], grads_fc[name])
        status = "OK" if err < 1e-7 else "FAIL"
        print(f"    {name} gradient rel_error: {err:.2e}  [{status}]")
        assert err < 1e-7, f"Gradient mismatch for {name}: rel_error={err:.2e}"

# ============================================================
# TEST 7: Deeper network gradient check (4 hidden layers)
# ============================================================
def test_deep_gradient_check():
    torch.manual_seed(42)
    N, D, C = 2, 10, 5
    hidden = [8, 6, 8, 6]
    X = torch.randn(N, D, dtype=torch.float64)
    y = torch.randint(C, (N,))

    model = FullyConnectedNet(hidden, input_dim=D, num_classes=C,
                              reg=0.5, weight_scale=5e-2, dtype=torch.float64)

    loss, grads = model.loss(X, y)
    print(f"  loss = {loss:.6f}")

    for name in sorted(grads):
        f = lambda _: model.loss(X, y)[0]
        grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
        error = rel_error(grad_num, grads[name])
        status = "OK" if error < 1e-4 else "FAIL"
        print(f"    {name} relative error: {error:.2e}  [{status}]")
        assert error < 1e-4, f"Gradient check failed for {name}: rel_error={error:.2e}"

# ============================================================
# TEST 8: Test mode (y=None) returns scores, train returns loss+grads
# ============================================================
def test_train_vs_test_mode():
    torch.manual_seed(42)
    N, D, C = 5, 20, 10
    hidden = [30]
    X = torch.randn(N, D, dtype=torch.float64)
    y = torch.randint(C, (N,))

    model = FullyConnectedNet(hidden, input_dim=D, num_classes=C,
                              weight_scale=1e-2, dtype=torch.float64)

    # Test mode
    scores = model.loss(X)
    assert isinstance(scores, torch.Tensor), "Test mode should return a tensor of scores"
    assert scores.shape == (N, C), f"Test scores shape: {scores.shape}"

    # Train mode
    result = model.loss(X, y)
    assert isinstance(result, tuple) and len(result) == 2, "Train mode should return (loss, grads)"
    loss, grads = result
    assert isinstance(grads, dict), "grads should be a dictionary"

    print(f"  Test mode scores shape: {scores.shape}")
    print(f"  Train mode loss: {loss:.6f}")
    print(f"  Train mode grads keys: {sorted(grads.keys())}")

# ============================================================
# TEST 9: Overfitting a tiny dataset
# ============================================================
def test_overfit_tiny():
    torch.manual_seed(42)
    N, D, C = 10, 20, 5
    hidden = [50, 30]
    X = torch.randn(N, D, dtype=torch.float64)
    y = torch.randint(C, (N,))

    model = FullyConnectedNet(hidden, input_dim=D, num_classes=C,
                              reg=0.0, weight_scale=1e-1, dtype=torch.float64)

    learning_rate = 1e-1
    losses = []
    for epoch in range(200):
        loss, grads = model.loss(X, y)
        losses.append(loss.item())
        for name in model.params:
            model.params[name] -= learning_rate * grads[name]

    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss:   {losses[-1]:.6f}")

    scores = model.loss(X)
    preds = scores.argmax(dim=1)
    accuracy = (preds == y).float().mean().item()
    print(f"  Training accuracy: {accuracy*100:.1f}%")

    assert losses[-1] < 0.1, f"Final loss {losses[-1]:.4f} is too high; network should overfit"
    assert accuracy > 0.9, f"Accuracy {accuracy*100:.1f}% is too low; network should overfit"


# ============================================================
# RUN ALL TESTS
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("FullyConnectedNet Verification Suite")
    print("=" * 60)

    run_test("1. Parameter initialization shapes", test_init_shapes)
    run_test("2. Forward pass output shape", test_forward_shape)
    run_test("3. Initial loss sanity check", test_initial_loss)
    run_test("4. Regularization increases loss", test_regularization_increases_loss)
    run_test("5. Gradient check (2 hidden layers)", test_gradient_check)
    run_test("6. Match TwoLayerNet (1 hidden layer)", test_matches_two_layer_net)
    run_test("7. Deep gradient check (4 hidden layers)", test_deep_gradient_check)
    run_test("8. Train vs test mode", test_train_vs_test_mode)
    run_test("9. Overfit tiny dataset", test_overfit_tiny)

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} passed, {failed} failed out of {passed+failed} tests")
    print(f"{'='*60}")
    if errors:
        print("\nFailed tests:")
        for name, err in errors:
            print(f"  - {name}: {err}")
