"""read a CMM program (Mistral, from Browne & Sharpe at SAO) file saved as text and convert measured points to a x,y,z file."""
def cmmconvert(filename,startline, step, outfile=None):
    l=open(filename,'r').readlines()
    c=[(a[0].split('<')[-1],a[1],a[2][:-1]) for a in [ll.split(',')[0:3] for ll in l [startline::step]]]
    if outfile is None: outfile=fn_add_subscript(filename,'_converted')
    open(outfile,'w').write('\n'.join(['\t'.join(x) for x in c]))
    

if __name__=='__main__':
    filename='measure_data/2014_11_21/01_OP2S06_surface.txt'
    startline=55
    step=6
    cmmconvert(filename, startline, step)