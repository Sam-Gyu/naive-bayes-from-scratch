import numpy as np

class MultinomialNB:
  def __init__(self, alpha = 1):
    self.alpha = alpha

  def counts_based_onclass(self, x, y):
    self.n_features = x.shape[1]
    self.count_matrix = []

    for j in range(self.n_classes):
        mask = y[:, j].astype(bool)
        class_counts = np.asarray(x[mask].sum(axis=0)).flatten()
        self.count_matrix.append(class_counts)

    self.class_count = y.sum(axis=0)


  def compute_priors(self):
    num = self.class_count
    total = self.class_count.sum()

    return num / total

  def compute_likelihood(self):
    matrix = np.array(self.count_matrix)  
    num = matrix + self.alpha
    total = num.sum(axis=1).reshape(-1, 1)
    return num / total

  def fit(self, X, y):
    if hasattr(X, 'sparse'):
        X = X.sparse.to_coo().tocsr()

    self.classes = np.unique(y)
    self.n_classes = len(self.classes)

    y_onehot = np.zeros((len(y), self.n_classes))
    for i, c in enumerate(self.classes):
      y_onehot[:, i] = (y == c)

    self.counts_based_onclass(X, y_onehot)
    self.priors = self.compute_priors()
    self.likelihoods = self.compute_likelihood()
  
  def predict_all(self, X_test):
    if hasattr(X_test, 'sparse'):
        X_test = X_test.sparse.to_coo().tocsr()
    
    def get_row(i):
        row = X_test.iloc[i]
        if hasattr(row, 'todense'):
            return np.asarray(row.todense()).flatten()
        return row.values
    
    return np.array([self.predict(get_row(i))
                     for i in range(X_test.shape[0])])

  def predict(self, point):
    probs = self.priors.copy()
    for i in range(self.n_features):
        cat = int(point[i])
        if cat == 1:
            probs *= self.likelihoods[:, i]
    return self.classes[np.argmax(probs)]