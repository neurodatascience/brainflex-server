"""Fed-BioMed training plan for tabular regression."""

import numpy as np
from declearn.optimizer.modules import ScaffoldClientModule
from declearn.optimizer.regularizers import FedProxRegularizer
from fedbiomed.common.datamanager import DataManager
from fedbiomed.common.dataset import TabularDataset
from fedbiomed.common.optimizers.optimizer import Optimizer
from fedbiomed.common.training_plans import FedSGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class TabularRegressorTrainingPlan(FedSGDRegressor):
    """Predict a numeric column from standardized feature columns."""

    # allowed metrics
    METRICS = ("MEAN_SQUARE_ERROR", "MEAN_ABSOLUTE_ERROR", "R2")
    DEFAULT_METRICS = ("MEAN_SQUARE_ERROR",)

    def init_dependencies(self):
        return [
            "import numpy as np",
            "from fedbiomed.common.datamanager import DataManager",
            "from fedbiomed.common.dataset import TabularDataset",
            "from fedbiomed.common.optimizers.optimizer import Optimizer",
            "from declearn.optimizer.modules import ScaffoldClientModule",
            "from declearn.optimizer.regularizers import FedProxRegularizer",
            "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score",
        ]

    def init_optimizer(self, optimizer_args):
        if "lr" not in optimizer_args:
            # use the default scikit-learn internal optimizer
            return None

        modules = [ScaffoldClientModule()] if optimizer_args.get("scaffold") else []
        regularizers = (
            [FedProxRegularizer(alpha=optimizer_args["fedprox_mu"])]
            if optimizer_args.get("fedprox_mu")
            else []
        )
        return Optimizer(
            lr=optimizer_args["lr"], modules=modules, regularizers=regularizers
        )

    def post_init(self, model_args, training_args, *args, **kwargs):
        super().post_init(model_args, training_args, *args, **kwargs)
        unknown = set(model_args.get("metrics", [])) - set(self.METRICS)
        if unknown:
            raise ValueError(
                f"Unknown metrics {sorted(unknown)}. Allowed metrics: {list(self.METRICS)}"
            )
        # start from the global mean of the target instead of 0
        if "target_mean" in model_args:
            self.model().intercept_ = np.array([float(model_args["target_mean"])])

    def set_data_loaders(self, train_data_loader, test_data_loader):
        if train_data_loader is None:
            # no training data when the node only tests the model
            # used in held-out node validation
            self.training_data_loader = None
            self.testing_data_loader = test_data_loader
        else:
            super().set_data_loaders(train_data_loader, test_data_loader)

    def training_data(self):
        model_args = self.model_args()
        columns = list(model_args["feature_columns"])
        mean = np.asarray(model_args["feature_mean"], dtype=float)
        std = np.asarray(model_args["feature_std"], dtype=float)
        if not len(columns) == len(mean) == len(std) == model_args["n_features"]:
            raise ValueError(
                "feature_columns, feature_mean and feature_std must all have"
                f" length n_features={model_args['n_features']}"
            )

        dataset = TabularDataset(
            input_columns=columns,
            target_columns=[model_args["target_column"]],
            transform=lambda x: (x - mean) / std,
        )
        return DataManager(dataset=dataset, shuffle=True)

    def testing_step(self, data, target):
        target = np.ravel(target)
        prediction = self.model().predict(data)
        metric_functions = {
            "MEAN_SQUARE_ERROR": mean_squared_error,
            "MEAN_ABSOLUTE_ERROR": mean_absolute_error,
            "R2": r2_score,
        }
        return {
            name: metric_functions[name](target, prediction)
            for name in self.model_args().get("metrics", self.DEFAULT_METRICS)
        }
