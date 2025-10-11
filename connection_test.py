import uhd 

print("Creating device...")
dlist = 
print("Found", dlist)
print("Creating device...")
usrp = uhd.usrp.MultiUSRP("type=x300,addr=192.168.30.2")
print("Tx rate now:", usrp.get_tx_rate(0))