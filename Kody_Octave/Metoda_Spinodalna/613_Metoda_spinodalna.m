%Inicjalizacja sta³ych i dostosowanie katalogu
clear;close all;clc; clear AADcsp1;
FS=14;
R=8.314;
pkg load windows
pkg load io

tablece = xlsread ('data.xlsx');

%% Wyci¹gnij ciœnienia, temperatury i same u
[rows,cols]=size(tablece)
rows=rows-1
cols=cols-1
ce(1,2:cols+1)=tablece(1,2:cols+1)
P=tablece(2:rows+1,1)
%P(1)=P(1)/10
P(1)=P(1)/1
P(2:end)=P(2:end)*1.0/1
%P(2:end)=P(2:end)*0.981/10
ce=tablece(2:rows+1,2:cols+1)
T=tablece(1,2:cols+1)

J=101;
bet=linspace(3,4,J);

%Procedura obliczania u metod¹ spinodaln¹
for n=1:cols;
    indx{n} = find(ce(:,n)>0);  
    pSun{n}=[(ce(indx{n},n)-ce(1,n)) (ce(indx{n},n)-ce(1,n)).^2 (ce(indx{n},n)-ce(1,n)).^3]\(P(indx{n})-P(1));
    for j=1:J;
       pspin(j)=[ce(indx{n},n).^bet(j)-ce(1,n)^bet(j)]\[P(indx{n})-P(1)]; 
       Pspin=pspin(j)*(ce(indx{n},n).^bet(j)-ce(1,n)^bet(j));
       mSpin(j)=mean(P(indx{n})-Pspin);
       vSpin(j)=std(P(indx{n})-Pspin);
    end
    [minv(n),indmin]=min(vSpin);
    minm(n)=mSpin(indmin);
    minbet(n)=bet(indmin);
    minpspin(n)=pspin(indmin);
    Pspin=minpspin(n)*(ce(indx{n},n).^minbet(n)-ce(1,n)^minbet(n));
end

% Tutaj uœredniono konkatenacjê do wektora minbeth.
% Dziêki temu mo¿emy wykonaæ dowoln¹ operacjê matematyczn¹ na
%minbetach, a program wrzuci j¹ do
%wektor o d³ugoœci n kolumn plus k dodatkowych funkcji (np. œrednia,
%mediana, jakaœ metryka itp.).mean_minbet = mean(minbet)

mean_minbet = mean(minbet)
minbet = [minbet mean_minbet] #Konkatenacja

#I modified that loop. 
#It no longer returns a single column but a set
  for n=1:cols;
    for k = 1:length(minbet);
     mpspin(n,k)=[ce(indx{n},n).^minbet(k)-ce(1,n)^minbet(k)]\[P(indx{n})-P(1)];
    end
  end 

%%%%
#subplot(3,1,[1 2])
for n=1:cols

  for k = 1:length(minbet);    
    csp(indx{n},n,k)=ce(1,n).*(1+(P(indx{n})-P(1))/mpspin(n,k)/ce(1,n)^minbet(n)).^(1/minbet(n));
  end  
    #plot(P(indx{n}),csp(indx{n},n),'.-','color',clr{n});
    Psndl(n)=minpspin(n)*ce(1,n)^minbet(n)-P(1);
end
xlabel('P, MPa')
ylabel('c, m/s')
title('Spinodal representation')
set(gca,'FontSize',FS)
subplot(3,1,3)

%
ylim([-0.5 0.75])
xlabel('P, MPa')
ylabel('\epsilon_c, %')
set(gca,'FontSize',FS)
print -dpng soundspinodal
%%%
%%%


for n=1:cols
%    pl=plot(P,ce(:,n),'o');
%    clr{n}=get(pl,'color');
%    hold on
   for k = 1:length(minbet);
    csp1(:,n,k)=ce(1,n).*(1+(P(:)-P(1))/mpspin(n,k)/ce(1,n)^minbet(k)).^(1/minbet(k));
%   plot(P,csp1(:,n),'.-','color',clr{n});
    Psndl1(n,k)=mpspin(n,k)*ce(1,n)^minbet(k)-P(1);
    end
end



ADcsp1=100*(ce-csp1)./ce;
for k = 1:length(minbet)
  AADcsp1(k)=mean(abs(ADcsp1(:,:,k)(~isnan(ADcsp1(:,:,k)))));
end
[Minimum, Index] = min(AADcsp1); #Return smallest value


## Ta czêœæ generuje tylko dane wyjœciowe i zapisuje csp1 do pliku
for k = 1:length(minbet)
 printf('AAD number: %i. is equal: %f. for power factor: %f.\n', k, AADcsp1(k), minbet(k))
end
printf('Smallest AAD value is: %f.\n', min(AADcsp1))
printf('\n\n')
printf('Best spinodal result is for power factor: %f. and Index equal: %i.\n',minbet(Index),Index)
printf('And best result is: \n')
printf('--------------------------------------------------')
best_spin = csp1(:,:,Index)

###Oblicz minimaln¹ rozbie¿noœæ dla ka¿dego wyniku ADCsp1
###i oblicz, który wiersz (ciœnienie) ma minimalne odchylenie
### Rozpoczynam procedurê od 2 rzêdu do koñca
for k = 1:length(minbet);
[maxval(k),row_max(k)] = max(max(abs(ADcsp1(:,:,k)),[],2));
end

###Calculate minimum discrepancy for each ADCsp1 result 
###and calculate which row (pressure) have minimum deviation
###I start procedure from 2nd row to end
for k = 1:length(minbet);
[minval(k),row_min(k)] = min(min(abs(ADcsp1(2:length(P),:,k)),[],2));
end


printf('\n\n')
printf('************************************************************************************************\n')
printf('**MD- max diecrepancy**MBT - power factor********P - pressure***********************************\n')
for k = 1:length(minbet)
 #printf('Max discrepancy is: %f for power factor: %f. and pressure P: %f\n', maxval(k), minbet(k), P(row_max(k)))
 printf('MD: %f \t | MBT: %f. \t | P: %f\n', maxval(k), minbet(k), P(row_max(k)))
end
printf('************************************************************************************************\n')
printf('\n\n')

printf('\n\n')
printf('************************************************************************************************\n')
printf('**MnD- min diecrepancy**MBT - power factor********P - pressure***********************************\n')
for k = 1:length(minbet)
 
 printf('MnD: %f \t | MBT: %f. \t | P: %f\n', minval(k), minbet(k), P(row_min(k)+1))
end
printf('************************************************************************************************\n')
printf('\n\n')


#Zapis surowych danych otrzymanych metod¹ spinodaln¹ do pliku txt
fid = fopen('best_cspin_data.txt', 'w+');%Save output to txt
for i=1:size(best_spin, 1)
    fprintf(fid, '%f ', best_spin(i,:));
    fprintf(fid, '\n');
end
fclose(fid);%Close
#-------------------------------------#
for k = 1:length(minbet)
  figure
	subplot(3,1,[1 2])
	for n=1:cols
		pl=plot(P,ce(:,n),'o');
		clr{n}=get(pl,'color');
		hold on
		plot(P,csp1(:,n,k),'.-','color',clr{n});
    #Psndl1(n)=mpspin(n)*ce(1,n)^mbet-P(1);
	end

  xlim([0 max(P)+0.03*(max(P))])
  xlabel('P, MPa')
  ylabel('c, m/s')
  title(sprintf('Spinodal representation (unified \gamma_a) minbet =  #%d', minbet(k)))
  set(gca,'FontSize',FS)
  subplot(3,1,3)
for n=1:cols
    plot(P,abs(ADcsp1(:,n,k)),'o--','color',clr{n});
    hold on
end
end	
#I created parts of 
#the code below so that the chart is always scaled up accordingly 

MAXIMUM = max(max(ADcsp1(:,:,Index)));
Range_Y1 = MAXIMUM+(MAXIMUM*0.2);
MINIMUM = min(min(ADcsp1(:,:,Index)));
Range_Y2 = MINIMUM+(MINIMUM*0.2);
xlim([0 max(P)+0.03*(max(P))])
ylim([Range_Y2 Range_Y1])
xlabel('P, MPa')
ylabel('\epsilon_c, %')
set(gca,'FontSize',FS)
