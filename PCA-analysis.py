import pandas as pd
import numpy as np
import itertools
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
import sys

useall = True

# Define the list of variable we will be reading from the data file
colors = ['purple','blue','cyan','green','yellow','orange','red','magenta','brown']

N_RPC = 4
RPC_data_type = ['IC', 'CURRENT', 'RATE']
RPC_types = ['RE2', 'RE4']
RPC_use = ['REF', 'IRR']
sep = '_'
data_labels = ['Current density', 'Gamma rate']

env_vars = ['envP', 'envT', 'envRH', 'gasT_in', 'gasT_out', 'gasRH_in', 'gasRH_out']
RPC_vars = [[],[]]

for dt in RPC_data_type:
    for i, rpc in  enumerate(RPC_types):
        element = dt + sep + rpc
        if dt != RPC_data_type[0]:
            for use in RPC_use:
                RPC_vars[i].append(element + sep + use)
        else:
            RPC_vars[i].append(element)

# Open and read data file
datafile = "Data.csv"
datafile_cols = env_vars + [item for sublist in RPC_vars for item in sublist]
print(datafile_cols)

df = pd.read_csv(datafile, names=datafile_cols, skiprows=1)

# Select the data of interest
env_to_use = []
env_labels = []
ylabels = []

if useall:
    env_to_use = env_vars
    env_labels = ['Env. Pres.', 'Bunker Temp.', 'Bunker air R.H.', 'Supply gas Temp.', 'Exhaust gas Temp.', 'Supply gas R.H.', 'Exhaust gas R.H.']
    ylabels = ['Atmospherical Pressure ($hPa$)',
               'Bunker Temperature ($^{\circ}C$)',
               'Bunker air Relative Humidity ($\%$)',
               'Supply gas Temperature ($^{\circ}C$)',
               'Exhaust gas Temperature ($^{\circ}C$)',
               'Supply gas Relative Humidity ($\%$)',
               'Exhaust gas Relative Humidity ($\%$)']
else :
    env_to_use = ['envP', 'gasT_in', 'gasT_out', 'gasRH_in', 'gasRH_out']
    env_labels = ['Env. Pres.', 'Supply gas Temp.', 'Exhaust gas Temp.', 'Supply gas R.H.', 'Exhaust gas R.H.']
    ylabels = ['Atmospherical Pressure ($hPa$)',
               'Supply gas Temperature ($^{\circ}C$)',
               'Exhaust gas Temperature ($^{\circ}C$)',
               'Supply gas Relative Humidity ($\%$)',
               'Exhaust gas Relative Humidity ($\%$)']

enviros = [df.loc[:, env].values for env in env_to_use]

datasets = [] # the actual data sets (tables) for each detector
# list of arrays for each rpc containing the integrated charge,
# density current, gamma rate values and gas flow values
rpcsets = [[],[],[]]

for rpc, use in itertools.product(RPC_types, RPC_use):
    PCA_columns = [] # columns to use
    element = rpc + sep + use
    for i, dt in enumerate(RPC_data_type):
        if dt != RPC_data_type[0]:
            col = dt+sep+element
            PCA_columns.append(col)
            rpcsets[i].append(df.loc[:, col].values)
        else:
            col = dt+sep+rpc
            rpcsets[i].append(df.loc[:, col].values)
    PCA_columns = PCA_columns + env_to_use
    datasets.append(df.loc[:, PCA_columns].values)

# PCA on the different data sets
N_PCs = len(env_to_use) + len(RPC_data_type) - 1
Cols = ['PC'+str(pc+1) for pc in range(N_PCs)]

PCAs = []
PCs = []
pDFs = []

scree, s_ax = plt.subplots(1, 1, figsize = (4,4))
bar_buffer = 0.1
bar_width = (1-bar_buffer)/N_RPC
tick_pos = np.arange(N_PCs)+(N_RPC-1)*bar_width/2.

for rpc in range(N_RPC):
    # Standardization of the original variables
    # The function shifts the distributions of each variables to obtain
    # an average of 0 and a variance of 1
    datasets[rpc] = StandardScaler().fit_transform(datasets[rpc])

    # Perform PCA
    PCAs.append(PCA(n_components=N_PCs, whiten=True))
    PCs.append(PCAs[rpc].fit_transform(datasets[rpc]))
    pDFs.append(pd.DataFrame(data = PCs[rpc], columns = Cols))

    # Plot the variability of each principal component (Scree plot)
    percent_variance = np.round(PCAs[rpc].explained_variance_ratio_* 100, decimals =2)
    bar_pos = np.arange(N_PCs)+rpc*bar_width
    rpc_label = RPC_types[int(rpc/len(RPC_types))]+' '+RPC_use[int(rpc%len(RPC_types))]
    s_ax.bar(x=bar_pos , height=percent_variance, width=bar_width, tick_label=Cols, label=rpc_label)

    # Print the covariance matrices and eigenvectors
    print(rpc_label)
    print('\nConvariance matrix\n%s\n' %np.cov(datasets[rpc].T))
    print('\nEigenvectors\n%s\n' %PCAs[rpc].components_)
    print('\nEigenvalues\n%s\n' %PCAs[rpc].explained_variance_)
    print('\nExplained variance\n%s\n' %(PCAs[rpc].explained_variance_ratio_*100))

s_ax.legend()
s_ax.grid()
s_ax.set_ylabel('Percentate of Variance Explained')
s_ax.set_xlabel('Principal Component')
s_ax.set_xticks(tick_pos)
scree.savefig('Scree_plot_Full-Data.png')
scree.savefig('Scree_plot_Full-Data.pdf')

# Make 3D plots with the 3 principal components containing the most
# information.
# Give a different color to the points according to the value of the
# current density J or of the gamma rate R.
# These values act as 4th dimension.
PC_plots = []
labels = [' Current density ($\mu A/cm^2$)', ' Gamma rate ($Hz/cm^2$)']
filenames = ['Data_separation_vs_Current_Density', 'Data_separation_vs_Gamma_Rate']

for p in range(2):
    PC_plots.append(plt.figure(figsize = (10,10)))
    for rpc in range(N_RPC):
        position = 221 + rpc
        PC_ax = PC_plots[p].add_subplot(position, projection='3d')
        PC_ax.set_xlabel('PC1')
        PC_ax.set_ylabel('PC2')
        PC_ax.set_zlabel('PC3')

        x = pDFs[rpc].loc[:, 'PC1']
        y = pDFs[rpc].loc[:, 'PC2']
        z = pDFs[rpc].loc[:, 'PC3']
        color = rpcsets[p+1][rpc]
        img = PC_ax.scatter(x, y, z, c = color, s = 50)
        cbar = PC_plots[p].colorbar(img)
        label = RPC_types[int(rpc/len(RPC_types))]+' '+RPC_use[int(rpc%len(RPC_types))]+labels[p]
        cbar.ax.set_ylabel(label, rotation = 270);
        PC_ax.grid()

    PC_plots[p].tight_layout()
    savePNG = filenames[p] + '.png'
    savePDF = filenames[p] + '.pdf'
    PC_plots[p].savefig(savePNG)
    PC_plots[p].savefig(savePDF)

# Make 3D plots with the 3 principal components containing the most
# information.
# Give a different color to the points according to the value of the
# environmental parameters
# These values act as 4th dimension.
PC_plots = []
filenames = []

if useall:
    filenames = ['Data_separation_vs_Pressure',
                 'Data_separation_vs_Bunker-Temperature',
                 'Data_separation_vs_Bunker-Humidity',
                 'Data_separation_vs_Gas-Temperature-IN',
                 'Data_separation_vs_Gas-Temperature-OUT',
                 'Data_separation_vs_Gas-Humidity-IN',
                 'Data_separation_vs_Gas-Humidity-OUT']
else :
    filenames = ['Data_separation_vs_Pressure',
                 'Data_separation_vs_Gas-Temperature-IN',
                 'Data_separation_vs_Gas-Temperature-OUT',
                 'Data_separation_vs_Gas-Humidity-IN',
                 'Data_separation_vs_Gas-Humidity-OUT']

for p in range(len(filenames)):
    PC_plots.append(plt.figure(figsize = (10,10)))
    for rpc in range(N_RPC):
        position = 221 + rpc
        PC_ax = PC_plots[p].add_subplot(position, projection='3d')
        PC_ax.set_xlabel('PC1')
        PC_ax.set_ylabel('PC2')
        PC_ax.set_zlabel('PC3')

        x = pDFs[rpc].loc[:, 'PC1']
        y = pDFs[rpc].loc[:, 'PC2']
        z = pDFs[rpc].loc[:, 'PC3']
        color = enviros[p]
        img = PC_ax.scatter(x, y, z, c = color, s = 50)
        cbar = PC_plots[p].colorbar(img)
        cbar.ax.set_ylabel(ylabels[p], rotation = 270);
        PC_ax.grid()
    PC_plots[p].tight_layout()
    savePNG = filenames[p] + '.png'
    savePDF = filenames[p] + '.pdf'
    PC_plots[p].savefig(savePNG)
    PC_plots[p].savefig(savePDF)

# Make 3D plots with the 3 principal components containing the most
# information.
# Give a different color to the points according to the value of the
# environmental parameters
# These values act as 4th dimension.
variables = data_labels + env_labels

Var_PCs, Var_PCs_ax = plt.subplots(2, 2, figsize = (6,6), sharex='col', sharey='row')

data_matrix = [[],[],[],[]]
env_matrix = [[],[],[],[]]

for rpc in range(N_RPC):
    position = 221 + rpc
    row = int(rpc/len(RPC_types))
    col = int(rpc%len(RPC_types))
    major = []

    if col == 0:
        Var_PCs_ax[row][col].set_ylabel('Principal Component')
    if row == 1:
        Var_PCs_ax[row][col].set_xlabel('Signal Strength')

    for i_var, var in enumerate(variables):
        x = []
        y = []
        for d in range(N_PCs):
            exp_var_value = PCAs[rpc].components_[d][i_var] * PCAs[rpc].explained_variance_[d]
            x.append(exp_var_value)
            y.append(d+1)
        major = y
        Var_PCs_ax[row][col].scatter(x, y, c=colors[i_var], label=var)
        if i_var < 2:
            data_matrix[rpc].append(x)
        else:
            env_matrix[rpc].append(x)

    Var_PCs_ax[row][col].set_xlim([-2,2])
    Var_PCs_ax[row][col].set_ylim([0.5,N_PCs*1.5+1])
    Var_PCs_ax[row][col].xaxis.set_major_locator(ticker.FixedLocator([-2,-1,0,1,2]))
    Var_PCs_ax[row][col].yaxis.set_major_locator(ticker.FixedLocator(major))
    Var_PCs_ax[row][col].grid()
    label = RPC_types[int(rpc/len(RPC_types))]+' '+RPC_use[int(rpc%len(RPC_types))]
    Var_PCs_ax[row][col].legend(loc='upper center', ncol=2, title=label, fontsize = 'xx-small')
Var_PCs.tight_layout()
Var_PCs.savefig('Full-Data-variation-Scores.png')
Var_PCs.savefig('Full-Data-variation-Scores.pdf')

# Print the result of the scalar product in the PC space in a
# latex tabular format
N_cols = len(data_labels)*N_RPC+1
output = open("latex-table.txt", "w+")
print('\\begin{tabular}{|',end="", file=output)
for c in range(N_cols):
    print('c|', end="", file=output)
print('}\n\\cline{2-%s}\n' % N_cols,end="", file=output)
print('\\multicolumn{1}{c|}{}',end="", file=output)
for d in data_labels:
    print(' & \\multicolumn{%s}{c|}{%s}' % (N_RPC,d), end="", file=output)
print('\\\\\n\\hline\n', end="", file=output)
print('Variables', end="", file=output)
for d in range(len(data_labels)):
    for rpc in range(N_RPC):
        rpc_label = RPC_types[int(rpc/len(RPC_types))]+' '+RPC_use[int(rpc%len(RPC_types))]
        print(' & %s' % rpc_label, end="", file=output)
print('\\\\\n\\hline\n', end="", file=output)
for i_e, env in enumerate(env_labels):
    print(env, end="", file=output)
    for i_d in range(len(data_labels)):
        for rpc in range(N_RPC):
            scalar = np.dot(data_matrix[rpc][i_d], env_matrix[rpc][i_e])
            print(' & %5.2f' % scalar, end="", file=output)
    print('\\\\\n\hline\n', end="", file=output)
print('\\end{tabular}', file=output)
output.close()
#plt.show()
