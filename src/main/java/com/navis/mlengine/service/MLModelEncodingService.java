package com.navis.mlengine.service;

import com.navis.mlengine.repository.IMLModelEncodingRepository;
import com.navis.mlengine.service.interfaces.IMLModelEncodingService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
public class MLModelEncodingService implements IMLModelEncodingService {

    private final IMLModelEncodingRepository modelEncodingRepository;
}