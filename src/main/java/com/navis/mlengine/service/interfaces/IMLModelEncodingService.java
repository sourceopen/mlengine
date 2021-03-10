package com.navis.mlengine.service.interfaces;

import com.navis.mlengine.entities.MLModelEncoding;

import java.util.List;

public interface IMLModelEncodingService {
    public List<MLModelEncoding> getAllEncodings();

    public List<MLModelEncoding> getAllEncodingsForConsumerId(String consumerId);
}
