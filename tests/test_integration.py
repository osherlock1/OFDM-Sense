import numpy as np
from ofdm.config import OFDMConfig
from ofdm.utils.generator import DataGenerator
from ofdm.core.preamble import generate_pilot_vals
from ofdm.core.payload import generate_ofdm_data_symbol, extract_data
from ofdm.core.waveform import create_time_domain_symbol, remove_cp, time_to_freq
from ofdm.modulation.qam import iq_to_binary
from ofdm.channel.noise import add_nosie
from ofdm.modulation.qam import iq_to_binary
from ofdm.utils.eval import calc_BER, calc_EVM, calc_SER
from ofdm.core.preamble import generate_pilot_symbol
from ofdm.channel.CHEST import channel_estimation_calc, apply_gains
from ofdm.channel.cfo import apply_cfo

def test_pipeline_no_channel_bits_recovered():
    """
    End-to-end: TX -> RX with ideal channel (no noise, no CFO, no delay).
    Decoded bits must exactly match transmitted bits
    """
    config = OFDMConfig()
    gen = DataGenerator(config=config, seed=0)
    iq_tx, bits_tx = gen.generate_random_payload()
    pilot_data = generate_pilot_vals(config=config)

    # ---tx---
    symbol_freq = generate_ofdm_data_symbol(config=config, iq_data=iq_tx, pilot_data=pilot_data)
    time_signal = create_time_domain_symbol(symbol_freq, cp_len=config.CP_LEN)

    # ---rx---
    time_no_cp = remove_cp(time_signal, cp_len=config.CP_LEN)
    rx_freq = time_to_freq(time_no_cp)
    iq_rx = extract_data(rx_freq, config)

    # --- decode ---
    scale = np.sqrt(10)
    bits_rx = "".join(iq_to_binary(sample * scale) for sample in iq_rx)

    assert bits_rx == bits_tx


def test_pipeline_high_snr_low_ber():
    """
    End-to-end with AGWN noise at high SNR
    BER should be very low but not guaranteeded to be zero.
    """
    config = OFDMConfig()
    gen = DataGenerator(config=config, seed=0)
    iq_tx, bits_tx = gen.generate_random_payload()
    pilot_data = generate_pilot_vals(config=config)

    # --- tx ---
    symbol_freq = generate_ofdm_data_symbol(config=config, iq_data=iq_tx, pilot_data=pilot_data)
    time_signal = create_time_domain_symbol(freq_data=symbol_freq, cp_len=config.CP_LEN)

    # --- add noise ---
    noisy_signal = add_nosie(signal=time_signal, snr_db=30.0, seed=42)

    # --- rx ---
    time_no_cp = remove_cp(noisy_signal, cp_len=config.CP_LEN)
    rx_freq = time_to_freq(time_no_cp)
    iq_rx = extract_data(rx_freq, config)

    # --- decode ---
    scale = np.sqrt(10)
    bits_rx = "".join(iq_to_binary(sample * scale) for sample in iq_rx)

    ber = calc_BER(
        iq_rx=np.array([complex(b) for b in bits_rx]),
        iq_ref=np.array([complex(b) for b in bits_tx])
    )    
    
    assert ber < 0.01

def test_pipeline_high_snr_low_ber():                                                                         
    config = OFDMConfig()
    gen = DataGenerator(config=config, seed=0)                                                                
    iq_tx, bits_tx = gen.generate_random_payload()                                                            
    pilot_data = generate_pilot_vals(config=config)                                                           
                                                                                                            
    # --- tx ---                                                                                              
    symbol_freq = generate_ofdm_data_symbol(config=config, iq_data=iq_tx, pilot_data=pilot_data)              
    time_signal = create_time_domain_symbol(symbol_freq, cp_len=config.CP_LEN)                                
                                                                                                            
    # --- nosie ---
    from ofdm.channel.noise import add_nosie                                                                  
    noisy_signal = add_nosie(signal=time_signal, snr_db=30.0, seed=42)

    # --- rx ---                                                                                              
    time_no_cp = remove_cp(noisy_signal, cp_len=config.CP_LEN)
    rx_freq = time_to_freq(time_no_cp)                                                                        
    iq_rx = extract_data(rx_freq, config)                                                                     
                                                                                                            
    # --- compare ---                                                                           
    scale = np.sqrt(10)                                                                                       
    bits_rx = "".join(iq_to_binary(sample * scale) for sample in iq_rx)                                       
                                                                                                            
    n_bit_errors = sum(b1 != b2 for b1, b2 in zip(bits_tx, bits_rx))                                          
    ber = n_bit_errors / len(bits_tx)                                                                         
                                                                                                            
    assert ber < 0.01

def test_pipeline_channel_gain_equalized_bits_recovered():                                                    
    """                                                                                                       
    End-to-end with a flat channel gain.                                                                      
    After channel estimation and equalization, decoded bits must exactly match TX.                            
    """                                                                                                       
    config = OFDMConfig()                                                                                     
    gen = DataGenerator(config=config, seed=0)                                                                
    iq_tx, bits_tx = gen.generate_random_payload()
    pilot_data = generate_pilot_vals(config=config)                                                           

    # --- tx ---                                                                                              
    pilot_symbol_freq = generate_pilot_symbol(config=config)
    pilot_time = create_time_domain_symbol(pilot_symbol_freq, cp_len=config.CP_LEN)                           
                                                                                                            
    data_symbol_freq = generate_ofdm_data_symbol(config=config, iq_data=iq_tx, pilot_data=pilot_data)         
    data_time = create_time_domain_symbol(data_symbol_freq, cp_len=config.CP_LEN)                             
                                                                                                            
    # --- flat gain ---
    channel_gain = 3.5 + 1.2j                                                                                 
    pilot_rx_time = pilot_time * channel_gain                                                                 
    data_rx_time  = data_time  * channel_gain                                                                 
                                                                                                            
    # --- rx ---                                                           
    pilot_rx_freq = time_to_freq(remove_cp(pilot_rx_time, cp_len=config.CP_LEN))
    Lambda = channel_estimation_calc(                                                                         
        rx_pilot_freq=pilot_rx_freq,
        tx_pilot_ref=pilot_symbol_freq,                                                                       
        config=config
    )                                                                                                         
                                                                           
    data_rx_freq = time_to_freq(remove_cp(data_rx_time, cp_len=config.CP_LEN))
    data_equalized = apply_gains(rx_symb_freq=data_rx_freq, Lambda_est=Lambda)                                
    iq_rx = extract_data(data_equalized, config)                                                              
                                                                                                            
    # --- decode ---                                                                                          
    scale = np.sqrt(10)
    bits_rx = "".join(iq_to_binary(sample * scale) for sample in iq_rx)                                       

    assert bits_rx == bits_tx   


def test_pipeline_cfo_corrected_bits_recovered():
    """
    End-to-end with a known CFO applied.
    After CFO correction, decoded bits must exactly match TX.
    """
    from ofdm.channel.cfo import apply_cfo

    config = OFDMConfig()
    gen = DataGenerator(config=config, seed=0)
    iq_tx, bits_tx = gen.generate_random_payload()
    pilot_data = generate_pilot_vals(config=config)

    # --- tx ---
    symbol_freq = generate_ofdm_data_symbol(config=config, iq_data=iq_tx, pilot_data=pilot_data)
    time_signal = create_time_domain_symbol(symbol_freq, cp_len=config.CP_LEN)

    # --- apply cfo ---
    cfo = 5000.0
    t = np.arange(len(time_signal)) / config.FS
    rx_signal = time_signal * np.exp(1j * 2 * np.pi * cfo * t)

    # --- correct cfo ---
    rx_corrected = apply_cfo(rx_signal=rx_signal, cfo=cfo, fs=config.FS)

    # --- rx ---
    time_no_cp = remove_cp(rx_corrected, cp_len=config.CP_LEN)
    rx_freq = time_to_freq(time_no_cp)
    iq_rx = extract_data(rx_freq, config)

    scale = np.sqrt(10)
    bits_rx = "".join(iq_to_binary(sample * scale) for sample in iq_rx)

    assert bits_rx == bits_tx