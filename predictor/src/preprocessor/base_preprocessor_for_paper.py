

class BasePredictor(ABC):
    def __init__(self, name, config):
        self.preprocessor = load_preprocessor(config)
        self.models = load_models(config)

    def predict(self, features, mode):
        results = []
        if mode == "structure":
            results += [m(features) for m in self.structure_models]
        elif mode == "runtime":
            results += [m(features) for m in self.runtime_models]

        results += [m(features) for m in self.common_models]
        
        return quantile_aggregate(results)

class BaseScheduler(ABC):
    def compute_plan(self, history):
        metrics = aggregate_metrics(history, group_by='model')
        next_plan = call_prediction_service(metrics, mode="structure")
        return next_plan

    @abstractmethod
    def policy(self, plan):
        # Policy of transform the plan to real deployment
        pass

class BaseRouter(ABC):
    def schedule(self, task, instances):
        scores = call_prediction_service(task, instances, mode="runtime")
        selected_instance = self._select_instance(instances, scores)
        return selected_instance

    @abstractmethod
    def _select_instance(self, instances, scores):
        # Select the best instance based on the scores
        pass