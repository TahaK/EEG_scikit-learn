import plot_compare_methods

# Author: Mustafa Taha Kocyigit -- <mustafataha93@gmail.com>

for i in range(0,10):
    plot_compare_methods.execut(range(30*i,30+i*30,1),'ch'+str(i)+'.png')
