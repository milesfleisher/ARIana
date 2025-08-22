import serial.tools.list_ports

print("Available serial ports:")
ports = serial.tools.list_list_ports.comports()
if not ports:
    print("  No serial ports found.")
for port in ports:
    print(f"  Port: {port.device}")
    print(f"    Description: {port.description}")
    print(f"    Hardware ID: {port.hwid}")
    print(f"    Manufacturer: {port.manufacturer}")
    print(f"    Product: {port.product}")
    print(f"    VID: {hex(port.vid) if port.vid else 'N/A'}")
    print(f"    PID: {hex(port.pid) if port.pid else 'N/A'}")
    print("\n")


