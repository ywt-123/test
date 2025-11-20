import h5py

filename = r'C:\Users\杨文韬\Desktop\论文\代码_新\final_weights.h5'
with h5py.File(filename, 'r') as f:
    def print_attrs(name, obj):
        print(name)

    f.visititems(print_attrs)