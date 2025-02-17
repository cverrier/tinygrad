from tinygrad.shape.shapetracker import ShapeTracker

a = ShapeTracker.from_shape((3, 2))
a = a.permute((1, 0))
a = a.reshape((3, 2))
idx, valid = a.to_indexed_uops()
print(idx.render())
