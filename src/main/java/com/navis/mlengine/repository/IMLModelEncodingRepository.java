package com.navis.mlengine.repository;

import com.navis.mlengine.entities.MLModelEncoding;
import org.springframework.data.repository.CrudRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface IMLModelEncodingRepository extends CrudRepository<MLModelEncoding, Long> {
}
