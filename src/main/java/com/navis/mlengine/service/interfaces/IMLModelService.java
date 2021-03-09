package com.navis.mlengine.service.interfaces;

import com.navis.mlengine.entities.MLModel;

import java.util.List;

public interface IMLModelService {
    public List<MLModel> getAllModels();
    public MLModel saveModel(MLModel model);
    public MLModel getModelForConsumer(String consumerId);
}

