from sklearn.linear_model import Lasso


clf = Lasso()
X_lasso = clf.fit(X=sim.C, y=sim.Y_fft)

plot_equation(sim.X_pred, sim.X, X_lasso, titles=("X_pred", "X", "X_lasso"), ratios=(1, 1, 1))
