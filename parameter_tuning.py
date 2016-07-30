import plot_compare_methods


for i in range(11,32):
    plot_compare_methods.execut(range(10*i,10+i*10,1),'ch'+str(i)+'.png')
