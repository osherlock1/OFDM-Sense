## eval_rx.py

Script used to evaluate the performance of each subcarrier the symbol.  

## generate_packet.py

Script used to generate OFDM Packets for transfer.  Specifify number of data symbols.  Add gaussian noise for testing.  Specify output file name.  

## mf_test.py

Test script for getting distance estimation from matched filter (mf) magnitude outputs.

## plot_rx_signal.py

Visualize the generated and recieved raw time series signals from USRP transfer

## run_transfer.py

Send generated data through USRP system.  Specifiy USRP parameters and comminication channel.

## test_sin_wave.py

Script to quickly check if USRP platform is set up correctly and succesfully transfering data.  Also calculates carrier frequency offset (CF)

## unpack_rx.py

Script that unpacks recieved OFDM pacckets and saves the recieved data to a json file.

## verifiy_sync_usrp.py

Script used to test the Schmidl-Cox syncronication algorithm's performance.