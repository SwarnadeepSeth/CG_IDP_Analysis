import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.stats import exponnorm
import scipy.stats as stats
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ================================================================================= #
# load data
df = pd.read_csv('charge.csv')
df.columns = ['Name', 'length', 'f', 'Q']

IDP_name = ['CspTm', 'R15', 'FhuA', 'An16', 'synuclein', 'sNase', 'ACTR', 'K32','OPN', 'SH4UD', 'CoINT', 'ProTaC', 'HST5', 'ProTaN']
colors =  ['red', 'green', 'teal', 'yellow', 'orange', 'purple', 'pink', 'brown', 'gray', 'black', 'white', 'cyan', 'magenta', 'blue']
eps = '0.1'

# ================================================================================= #
print ("-"*50)
print ("Routine to plot histogram using Seaborn")
print ("-"*50)
print ("Choose the variable for the histogram:")
print ("1. <Rg^2>, 2. sqrt(<Rg^2}, 3. <Rend^2> 4. sqrt(<Rend^2>), 5. <Trans_Fluct^2>, 6. sqrt(Trans_Fluct^2) \n")
data_index = int(input("Enter the index: "))

data_dict = {
        "1": [3, r'$\bar{R}_g^2$', r'$P(\bar{R}_g^2)$', 'Rg2'],
        "2": [2, r'$\bar{R}_g$', r'$P(\bar{R}_g)$', 'Rg'],
        "3": [4, r'$\bar{R}_{end}^2$', r'$P(\bar{R}_{end}^2$)', 'Rend2'],
        "4": [4, r'$\bar{R}_{end}$', r'$P(\bar{R}_{end})$', 'Rend'],
        "5": [5, r'$\bar{TF}^2$', r'$P(\bar{TF}^2)$', 'TF2' ],
        "6": [5, r'$\bar{TF}$', r'$P(\bar{TF})$', 'TF']
    }

index = data_dict[str(data_index)][0]
xlabel = data_dict[str(data_index)][1]
ylabel = data_dict[str(data_index)][2]
output_label = data_dict[str(data_index)][3]

print ("You have chosen:", xlabel)

# ================================================================================= #
fig, ax = plt.subplots(figsize=(10, 6))

#df.sort_values(by=['f'], inplace=True)
i=0
for IDP in df['Name']:
#for IDP in IDP_name:

    # check if the running_stat.dat file exists
    try:
        data = np.loadtxt(IDP + f"/E_{eps}/running_stat.dat", usecols=(data_index), dtype=float, unpack=True)
    except:
        print (f"running_stat.dat file does not exist for {IDP}")
        continue

    if data_index == 4 or data_index == 6:
        data_arr = np.sqrt(data)
    else:
        data_arr = data

    data_avg = np.mean(data_arr)
    # sacled data
    data_arr = data_arr/data_avg

    # ================================================================================= #
    # Plot the histogram.
    # fit kwargs color same as the histogram
    #ax = sns.distplot(data_arr, kde=False, color=color, fit=exponnorm, fit_kws={"color":color}, label=IDP)
    #ax = sns.distplot(data_arr, kde=False, fit=exponnorm, color=color, label=IDP)

    # Get the fitted parameters used by sns
    (K, loc, l) = exponnorm.fit(data_arr)
    print ("-"*50)
    print ("IDP:", IDP)
    print ("The Parmaters of the Fitted Histogram:")
    print ("K={0}, loc={1}, scale={2}".format(K, loc, l))

    # statistics
    mean, var, skew, kurt = exponnorm.stats(K, moments='mvsk')
    print ("mean:", mean, "var:", var, "skew:", skew, "kurt:", kurt)

    print ("Skewness:", 2*K**3/(1+K**2)**(3/2))

    # append the skewness to the dataframe
    df.loc[df['Name'] == IDP, 'skewness'] = skew

    # Cross-check this is indeed the case - should be overlaid over black curve
    x_dummy = np.linspace(min(data_arr), max(data_arr), 100)
    y_fit = exponnorm.pdf(x_dummy, K, loc, l)

    legend_label = IDP + ' (' + str(np.round(skew,2))+')'

    # ================================================================================= #
    if IDP in IDP_name:
        if skew > 0.03:
            ax.plot(x_dummy, y_fit, color=colors[i], linewidth=3, label=legend_label)
            i += 1
        else:
            ax.plot(x_dummy, y_fit, color=colors[i], linewidth=3, linestyle='--', label=legend_label)
            i += 1

    # Maximum of the Histogram Fit
    y_fit_max = round(y_fit.max(),2)
    x_fit_max = round(x_dummy[np.argmax(y_fit)],2)
    print ("-"*50)
    print ("X-max of Fitted Histogram:", x_fit_max)
    print ("Y-max of Fitted Histogram:", y_fit_max)

    # ================================================================================= #

# draw a vline at x=1
ax.axvline(x=1, color='black', linewidth=2)

plt.xlabel(f'{xlabel}', color='darkgreen', fontsize=35)
plt.ylabel(f'{ylabel}', color='darkgreen', fontsize=35)
plt.xticks(fontsize=30, color='blue')
plt.yticks(fontsize=30, color='blue')
plt.legend(loc='upper right', frameon=False, fontsize=15)

plt.ylim(0,)

plt.tight_layout()
#plt.savefig(f"{output_label}_histogram.pdf")
plt.show()

# ========================================================================================= #
# load the symbol and color data
df_color = pd.read_csv('symbol_color.csv')
df_color.columns = ['Name', 'Symbol', 'Color']

# merge with the df by Name index
df = pd.merge(df, df_color, on='Name')
# ================================================================================= #

df_skew = df.dropna()
# sort the dataframe by the f value and reset the index
df_skew = df_skew.sort_values(by=['f'])
df_skew = df_skew.reset_index(drop=True)
print (df_skew)

plt.figure(figsize=(10,6))

# put two horizontal line at y=0.5 and y=-1.0
plt.axhline(y=0.5, color='black', linewidth=1.5)
plt.axhline(y=1.0, color='black', linewidth=1.5)

# shade the regions below y=0.5 as blue, between y=0.5 to y=1.0 green and above y=1.0 as red with alpha=0.1
plt.axhspan(-0.05, 0.5, facecolor='blue', alpha=0.1)
plt.axhspan(0.5, 1.0, facecolor='green', alpha=0.1)
plt.axhspan(1.0, 1.8, facecolor='red', alpha=0.1)


# plot the data
for i in range(len(df_skew['Name'])):
    # plt hollow symbols
    plt.scatter(df_skew['f'][i], df_skew['skewness'][i], marker=df_skew['Symbol'][i], edgecolor=df_skew['Color'][i], facecolor='none', s=150, lw=4, label=df_skew['Name'][i])
    print (df_skew['Name'][i], df_skew['Symbol'][i], df_skew['Color'][i])
# annotate the points
for i, txt in enumerate(df_skew['Name']):
    try:
        plt.annotate(txt, (df_skew['f'][i], df_skew['skewness'][i]), fontsize=15)
    except:
        print (txt)
        continue

# put the legend outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False, fontsize=10)

# save df_skew['f'], df_skew['skewness'] in a dat file
#df_skew.to_csv('skewness_vs_f.dat', sep='\t', index=False)

# draw a line with function 1.25/(x+1)**12
x = np.linspace(0, 0.28, 100)
y = 1.25/(x+1)**12
#plt.plot(x, y, color='black', linewidth=3, linestyle='--')

# draw a line with 1.55*exp(-17.23*x)
x = np.linspace(0, 0.28, 100)
y = 1.5*np.exp(-21*x)
plt.plot(x, y, color='blue', linewidth=3)

# =====================================================================================================
# fit the data by a line
slope, intercept, r_value, p_value, std_err = stats.linregress(df_skew['f'], df_skew['skewness'])
slope = round(slope, 2)
#intercept = round(intercept, 2)
#print ("slope:", slope, "intercept:", intercept, "r_value:", r_value, "p_value:", p_value, "std_err:", std_err)

# plot the line
x = np.linspace(0, 0.28, 100)
y = slope*x + intercept
#plt.plot(x, y, color='green', linewidth=3, linestyle='--')
#plt.text(0.1, 1.5, r'$\mathcal{S}$='+str(slope)+'$f^{*}+$'+str(intercept).format(slope, intercept), fontsize=15)

# -------------------------------------------------------------------------------------
df_skew_low = df_skew[df_skew['f'] < 0.06]

# fit the data by a line
slope, intercept, r_value, p_value, std_err = stats.linregress(df_skew_low['f'], df_skew_low['skewness'])
slope = round(slope, 2)
intercept = round(intercept, 2)
print ("slope:", slope, "intercept:", intercept, "r_value:", r_value, "p_value:", p_value, "std_err:", std_err)

# plot the line
x = np.linspace(-0.01, 0.1, 100)
y = slope*x + intercept
#plt.plot(x, y, color='red', linewidth=3, linestyle='--')
#plt.text(0.1, 1.5, r'$\mathcal{S}$='+str(slope)+'$f^{*}+$'+str(intercept).format(slope, intercept), fontsize=15)

# -------------------------------------------------------------------------------------
df_skew_high = df_skew[df_skew['f'] > 0.06]

# fit the data by a line
slope, intercept, r_value, p_value, std_err = stats.linregress(df_skew_high['f'], df_skew_high['skewness'])
slope = round(slope, 2)
intercept = round(intercept, 2)
print ("slope:", slope, "intercept:", intercept, "r_value:", r_value, "p_value:", p_value, "std_err:", std_err)

# plot the line
x = np.linspace(0.12, 0.28, 100)
y = slope*x + intercept
#plt.plot(x, y, color='red', linewidth=3, linestyle='--')
#plt.text(0.1, 1.5, r'$\mathcal{S}$='+str(slope)+'$f^{*}+$'+str(intercept).format(slope, intercept), fontsize=15)

# =========================================================================================================

plt.ylim(-0.05, 1.8)

plt.xlabel(r'$f^{*}$', fontsize=35)
plt.ylabel(r'$\mathcal{S}$', fontsize=35)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.tight_layout()
#plt.legend(loc='upper right', frameon=False, fontsize=25)
plt.savefig('skewness_vs_f_new.pdf')
plt.show()

















