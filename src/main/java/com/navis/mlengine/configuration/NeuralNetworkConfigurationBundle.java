package com.navis.mlengine.configuration;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Setter
@AllArgsConstructor
@Getter
@NoArgsConstructor
public class NeuralNetworkConfigurationBundle extends GenericAlgorithmConfigurationBundle {
    private int verbosity;
}
