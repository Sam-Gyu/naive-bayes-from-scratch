import numpy as np

class MultinomialNB:
  def __init__(self, alpha = 1):
    self.alpha = alpha

  def counts_based_onclass(self, x, y):
    self.n_features = x.shape[1]
    self.count_matrix = []

    for i in range(self.n_features):
      X_feature = x[:, i]
      unique_values = np.unique(X_feature)
      n_categories = int(max(unique_values.max(), 0)) + 1
      count_feature = np.zeros((self.n_classes, n_categories))

      for j in range(self.n_classes):
        mask = y[:, j].astype(bool)
        counts = np.bincount(X_feature[mask], minlength=n_categories)
        count_feature[j] = counts

      self.count_matrix.append(count_feature)

    self.class_count = y.sum(axis=0)


  def compute_priors(self, class_count):
    num = class_count
    total = class_count.sum()

    return num / total

  def compute_likelihood(self, count_matrix, n_features):
    likelihoods = []

    for i in range(n_features):
      num = count_matrix[i] + self.alpha
      total = num.sum(axis=1).reshape(-1,1)
      prob = num / total
      likelihoods.append(prob)

    return likelihoods


  def fit(self, X, y):
    if hasattr(X, 'to_numpy'):
        X = X.to_numpy()

    self.classes = np.unique(y)
    self.n_classes = len(self.classes)

    y_onehot = np.zeros((len(y), self.n_classes))
    for i, c in enumerate(self.classes):
      y_onehot[:, i] = (y == c)

    self.counts_based_onclass(X, y_onehot)
    self.priors = self.compute_priors(self.class_count)
    self.likelihoods = self.compute_likelihood(self.count_matrix, self.n_features)

  def predict_all(self, X_test):
    if hasattr(X_test, 'to_numpy'):
            X_test = X_test.to_numpy()

    return np.array([self.predict(point)
                    for point in X_test])

  def predict(self, point):
    probs = np.zeros(self.n_classes)

    for i in range(self.n_features):
      cat = int(point[i])
      max_cat = self.likelihoods[i].shape[1] - 1
      cat = min(cat, max_cat)
      probs += self.likelihoods[i][:, cat]

    probs += self.priors

    return self.classes[np.argmax(probs)]