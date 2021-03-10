package com.navis.mlengine.service;

import com.navis.mlengine.entities.MLModelEncoding;
import com.navis.mlengine.repository.IMLModelEncodingRepository;
import com.navis.mlengine.service.interfaces.IMLModelEncodingService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Service
@RequiredArgsConstructor
public class MLModelEncodingService implements IMLModelEncodingService {

    private final IMLModelEncodingRepository modelEncodingRepository;

    @Override
    @Transactional
    public List<MLModelEncoding> getAllEncodings() {
        return modelEncodingRepository.findAll();
    }

    @Override
    @Transactional
    public List<MLModelEncoding> getAllEncodingsForConsumerId(String consumerId) {
        return modelEncodingRepository.findAll();
    }
}