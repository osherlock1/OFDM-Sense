from ofdm_manager import OFDMManager
from subcarrier_map import SubcarrierMap
decimal = 3
binary = format(decimal, "04b")
print(type(binary))
map = SubcarrierMap()
om = OFDMManager(map)


#om.binary_to_iq(binary)

iq_samples = []

numbers = [1, 2, 3, 5, 2, 3, 6, 8]
for num in numbers:
    binary = format(num, "04b")
    iq_samples.append(om.binary_to_iq(binary))

print(iq_samples)


outputs = []
for iq in iq_samples:
    binary = om.iq_to_binary(iq)
    print(binary)