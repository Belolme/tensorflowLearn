def raise_exception():
    raise Exception()

try:
    raise_exception()
except Exception:
    print("catch exception")
else:
    print("everthing is normal")
finally:
    print("finally called")
