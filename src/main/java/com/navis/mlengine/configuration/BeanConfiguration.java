package com.navis.mlengine.configuration;

import com.navis.mlengine.algorithms.XGBoost.XGBoostImplementationWorkflow;
import com.navis.mlengine.mlhelpers.encoders.Encoders;
import com.sun.org.apache.bcel.internal.generic.INSTANCEOF;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.ApplicationContext;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Scope;

@Configuration
public class BeanConfiguration {

    private static BeanConfiguration INSTANCE = null;

    public static BeanConfiguration beanConfiguration() {
        if(INSTANCE ==null)
            INSTANCE = new BeanConfiguration();
        return INSTANCE;
    }

    @Autowired
    private ApplicationContext context;

    @Bean(name = "xgboostimplementationworkflow")
    @Scope("prototype")
    public XGBoostImplementationWorkflow getXGBoostImplementationWorkflow() {
       return new XGBoostImplementationWorkflow();
    }

    @Bean(name = "encoder")
    @Scope("prototype")
    public Encoders getEncoder() {
        return new Encoders();
    }

    @Bean(name = "xgboostconfigurationbundle")
    @Scope("prototype")
    public XGBoostConfigurationBundle getXGBoostConfigurationBundle() {
        return new XGBoostConfigurationBundle();
    }

}
