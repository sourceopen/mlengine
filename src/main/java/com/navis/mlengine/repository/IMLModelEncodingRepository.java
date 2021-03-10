package com.navis.mlengine.repository;

import com.navis.mlengine.entities.MLModelEncoding;
import org.springframework.data.repository.CrudRepository;
import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Repository
public interface IMLModelEncodingRepository extends CrudRepository<MLModelEncoding, Long> {
    @Transactional
    List<MLModelEncoding> findAll();

    @Transactional
    List<MLModelEncoding> findAllByConsumerIdIs(String consumerId);
}
