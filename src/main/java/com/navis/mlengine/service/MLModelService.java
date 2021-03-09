package com.navis.mlengine.service;

import com.navis.mlengine.entities.MLModel;
import com.navis.mlengine.repository.IMLModelRepository;
import com.navis.mlengine.service.interfaces.IMLModelService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Service
@RequiredArgsConstructor
public class MLModelService implements IMLModelService {

    private final IMLModelRepository modelRepository;

    @Override
    @Transactional
    public List<MLModel> getAllModels() {
        return modelRepository.findAll();
    }

    @Override
    @Transactional
    public MLModel getModelForConsumer(String consumerId) {
        return modelRepository.findMLModelByConsumerIdIs(consumerId);
    }

    @Override
    @Transactional
    public MLModel saveModel(MLModel model) {
        return modelRepository.save(model);
    }
}
