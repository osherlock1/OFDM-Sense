from ofdm_manager import OFDMManager
from subcarrier_map import SubcarrierMap
decimal = 3
binary = format(decimal, "04b")
print(type(binary))
map = SubcarrierMap()
om = OFDMManager(map)


om.binary_to_iq(binary)

