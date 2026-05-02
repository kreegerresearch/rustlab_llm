# Script:  overfit_demo.r
# System:  Same embedding+head bigram model, but on a corpus too small to
#          generalise.  Trains until train loss is near zero (memorisation)
#          while held-out val loss bottoms out and rises.
# Concept: When a model has more parameters than the data can constrain it
#          memorises the training pairs and val loss starts climbing again.
#          Visible signature: train ↘, val ↗.
# Equations: same forward / backward as train_loop.r
# Units:    parameters dimensionless; loss in nats.

seed(180);
vocab = 3;
d_emb = 16;     # bigger than train_loop.r — more capacity to memorise

E = randn(vocab, d_emb) * 0.3;
W = randn(d_emb, vocab) * 0.3;

# === Tiny corpus: a deterministic 6-element training set ===
# Train pairs: (1,2), (2,3), (3,2), (2,1), (1,2)  — the model can memorise this
# Val pairs:   different transitions to expose overfitting
train_pairs_curr = [1, 2, 3, 2, 1];
train_pairs_next = [2, 3, 2, 1, 2];
val_pairs_curr   = [3, 2, 1];
val_pairs_next   = [2, 1, 3];        # last pair (1,3) is rare in train

n_train_pairs = 5;
n_val_pairs = 3;

# === Forward + backward (struct return — see lesson 18 train_loop.r) ===
function r = step_grad(curr, nxt, E, W, vocab, d_emb)
  h_m = E(curr);
  p = softmax(h_m * W);
  L_step = -log(p(nxt));
  e_y = zeros(vocab); e_y(nxt) = 1.0;
  dlogits = p - e_y;
  dW_acc = h_m' * dlogits;
  dh = dlogits * W';
  dE_acc = zeros(vocab, d_emb);
  for k = 1:d_emb
    dE_acc(curr, k) = dh(k);
  end
  r = struct("dE", dE_acc, "dW", dW_acc, "L", L_step);
end

function L = mean_loss_pairs(currs, nexts, n_pairs, E, W)
  L = 0.0;
  for k = 1:n_pairs
    curr = currs(k);
    nxt  = nexts(k);
    h_m = E(curr);
    p = softmax(h_m * W);
    L = L + (-log(p(nxt)));
  end
  L = L / n_pairs;
end

# === Hyperparameters — push hard, no decay ===
n_train = 800;
beta1 = 0.9;
beta2 = 0.999;
adam_eps = 1.0e-8;
eta = 0.05;     # constant LR — no schedule, no decay
lambda_wd = 0.0;

m_E = zeros(vocab, d_emb);
v_E = zeros(vocab, d_emb);
m_W = zeros(d_emb, vocab);
v_W = zeros(d_emb, vocab);

loss_train = zeros(n_train + 1);
loss_val   = zeros(n_train + 1);
loss_train(1) = mean_loss_pairs(train_pairs_curr, train_pairs_next, n_train_pairs, E, W);
loss_val(1)   = mean_loss_pairs(val_pairs_curr,   val_pairs_next,   n_val_pairs,   E, W);

print("Initial train loss:", loss_train(1));
print("Initial val   loss:", loss_val(1));

for t = 1:n_train
  dE_sum = zeros(vocab, d_emb);
  dW_sum = zeros(d_emb, vocab);
  for k = 1:n_train_pairs
    curr = train_pairs_curr(k);
    nxt  = train_pairs_next(k);
    r = step_grad(curr, nxt, E, W, vocab, d_emb);
    dE_sum = dE_sum + r.dE;
    dW_sum = dW_sum + r.dW;
  end
  dE_avg = dE_sum / n_train_pairs;
  dW_avg = dW_sum / n_train_pairs;

  m_E = beta1 * m_E + (1 - beta1) * dE_avg;
  v_E = beta2 * v_E + (1 - beta2) * (dE_avg .^ 2);
  E = E - eta * (m_E / (1 - beta1 ^ t)) ./ (sqrt(v_E / (1 - beta2 ^ t)) + adam_eps);

  m_W = beta1 * m_W + (1 - beta1) * dW_avg;
  v_W = beta2 * v_W + (1 - beta2) * (dW_avg .^ 2);
  W = W - eta * (m_W / (1 - beta1 ^ t)) ./ (sqrt(v_W / (1 - beta2 ^ t)) + adam_eps);

  loss_train(t + 1) = mean_loss_pairs(train_pairs_curr, train_pairs_next, n_train_pairs, E, W);
  loss_val(t + 1)   = mean_loss_pairs(val_pairs_curr,   val_pairs_next,   n_val_pairs,   E, W);
end

print("Final train loss:", loss_train(n_train + 1));
print("Final val   loss:", loss_val(n_train + 1));
print("Train loss should be ~0 (memorised); val loss should rise above its initial value.");

# === Plot ===
figure()
steps = 0:n_train;
plot(steps, loss_train, "color", "red",  "label", "train")
hold("on")
plot(steps, loss_val,   "color", "blue", "label", "val")
hold("off")
title("Overfit signature:  train falls to ~0, val bottoms then rises")
xlabel("step")
ylabel("L (nats)")
legend("train", "val")
savefig("overfit_demo.svg")
print("Saved overfit_demo.svg");
