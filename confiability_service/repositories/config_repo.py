class ConfigRepository:
    def thresholds(self):
        return {
            "public": 0.25,
            "internal": 0.5,
            "confidential": 0.75,
            "restricted": 0.9
        }

    def class_order(self):
        # for mapping probabilities
        return ["public", "internal", "confidential", "restricted"]