import paramiko
import pickle
import numpy as np
import os 

def get_shot(archive, ndict, shots, host = 'cat1', user = 'kolesnichenko', secret = 'Ujyujkcnfqk(1)', port = 22):
    """ return package from archive from kusts in b dicts and numbers in shots
    format of package {'kust1': [(array, tstart, tstop, N_elements, name_signal, shot number), ...],
    'kust2': [(array, tstart, tstop, N_elements, name_signal, shot number), ...} """
    save_dir = "data/"+archive.split('/')[-1][:-5]
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    localpath = 'h.npy'
    remotepath = '/home/kolesnichenko/Progs/PyPack/h.npy'
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(hostname=host, username=user, password=secret, port=port)
    except:
        print("CAN'T ACCESS SERVER")
    rootpackage = dict()
    for kust in ndict.keys():
        # print(kust)
        for sign in ndict[kust]:
            for shot in shots:
                if not os.path.isfile(save_dir+'/'+str(shot)+'/'+kust+'/'+sign+'.pkl'):
                    py_command = "\"import Trans_ROOT; Trans_ROOT.F_open('" + archive + "', '" + \
                    kust + "', '" + sign + "', " + str(shot) + ")\""
                    com = "cd Progs/PyPack; python -c " + py_command
                    stdin, stdout, stderr = client.exec_command(com)
                    data = stdout.read() + stderr.read()
                    if data.decode('utf-8').replace("\n", "") != '0':
                        print(data.decode('utf-8'))
                    sftp = client.open_sftp()
                    sftp.get(remotepath, localpath)
                    h = np.load(localpath)
                    N = (len(h) - 3)
                    tstart = h[1]
                    tstop = tstart + h[0]*N
                    real_shot = int(h[2])
                    h = h[3:]
                    # saving data
                    if not os.path.isdir(save_dir+'/'+str(real_shot)):
                        os.mkdir(save_dir+'/'+str(real_shot))
                    if not os.path.isdir(save_dir+'/'+str(real_shot)+'/'+kust):
                        os.mkdir(save_dir+'/'+str(real_shot)+'/'+kust)
                    if not os.path.isdir(save_dir+'/'+str(real_shot)+'/'+kust+'/'):
                        os.mkdir(save_dir+'/'+str(real_shot)+'/'+kust)
                    with open(save_dir+'/'+str(real_shot)+'/'+kust+'/'+sign+'.pkl', 'wb') as f:
                        pickle.dump((h, tstart, tstop, N), f)
                else:
                    with open(save_dir+'/'+str(shot)+'/'+kust+'/'+sign+'.pkl', 'rb') as f:
                        h, tstart, tstop, N = pickle.load(f)
                        real_shot = shot
                # packing of rootpackage
                sending = (h, tstart, tstop, N, sign, real_shot)
                if rootpackage.get(kust):
                    rootpackage[kust].append(sending)
                else:
                    rootpackage[kust] = [sending]
    client.close()
    if os.path.isfile(localpath): 
        os.remove(localpath)
    # else: 
    #     print("File doesn't exists!")
    return rootpackage

if __name__ == '__main__':
    ArchiveName = "/home/murakhtin/test.root"
    names = {"Vacuum": ["TeProbeMirror", "neProbeMirror", "PotentialProbeMirror",
                "TeProbePG", "neProbePG", "PotentialProbePG"]}
    shots = [170, 171]
    get_shot(ArchiveName, names, shots)