package com.navis.mlengine.repository;

import com.navis.mlengine.entities.MLModel;
import org.springframework.data.repository.CrudRepository;
import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Repository
public interface IMLModelRepository extends CrudRepository<MLModel, Long> {
    @Transactional
    List<MLModel> findAll();

    @Transactional
    MLModel findMLModelByConsumerIdIs(String consumerId);
}

