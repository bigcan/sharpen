
import vectorbt as vbt
print(f"VectorBT Version: {vbt.__version__}")
print("Portfolio Attributes:")
print([x for x in dir(vbt.Portfolio) if 'from' in x])

print("\nChecking for Returns accessor:")
if hasattr(vbt, 'Returns'):
    print("vbt.Returns exists")
    print([x for x in dir(vbt.Returns) if not x.startswith('_')])
else:
    print("vbt.Returns DOES NOT exist")

print("\nChecking for Generic Accessors:")
print([x for x in dir(vbt) if 'Returns' in x or 'accessors' in x])
