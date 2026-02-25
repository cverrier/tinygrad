from tinygrad import Tensor

t = Tensor([1,2,3,4])
print(t.uop)
print("="*80)
t.realize()
print(t.uop)
