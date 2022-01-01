import matplotlib.pyplot as plt

from ..strain import total_plastic_strain_percent, total_elastic_strain_percent

def test_strain_vals(data):
    for test in data.data:
        pl, el = total_plastic_strain_percent(test), total_elastic_strain_percent(test)
        print(('Sample: %s'%test.Sample).ljust(15), 'Strain: %.2f'%test.Strain,\
              'Plastic: %.3f'%pl, 'Elastic: %.3f'%el, sep = '  ', end = '  ')
        print('Error: %.3f'%(abs(pl+el-test.Strain)/test.Strain*100))
    

