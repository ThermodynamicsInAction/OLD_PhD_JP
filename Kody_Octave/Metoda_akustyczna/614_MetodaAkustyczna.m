clear;close all;clc

pkg load windows
pkg load io
%TEN PLIK CZYTA EKSPERYMENTALNE WARTO�CI U oraz Rho (Atm)
FS=14;
R=8.314;
% Wybierz plik Excel z okna dialogowego
[filename, filepath] = uigetfile('results.xls', 'Wybierz plik Excel');

% Sprawd�, czy u�ytkownik wybra� plik
if isequal(filename, 0)
    disp('Anulowano wyb�r pliku. Skrypt zostanie zako�czony.');
    return;
end
% Skomponuj pe�n� �cie�k� do pliku
fullpath = fullfile(filepath, filename);

% Za�aduj pojemno�� ciepln�
TCp = xlsread(fullpath, 'ATMNN', 'D2:D22');
Cp = xlsread(fullpath, 'ATMNN', 'E2:E22') * 1e0;
[coefcp{1}, ~, coefcp{2}] = polyfit(TCp, Cp, 2);
Cps = @(T) polyval(coefcp{1}, T, [], coefcp{2});
 
#Za�aduj g�sto�� pod p=atm.
#W zale�no�ci od podej�cia za�aduj g�sto�ci eksperymentalne
#lub otrzymane za pomoc� metody uczenia m.

Td = xlsread (fullpath, 'ATMNN', 'A2:A22');
d = xlsread(fullpath, 'ATMNN', 'B2:B22'); 
[coefrho{1},~,coefrho{2}]=polyfit(Td,d,2);
rhos=@(T) polyval(coefrho{1},T,[],coefrho{2});

#Za�aduj tabel� pr�dko�ci d�wi�ku.
#W zale�no�ci od testowanej metody za�aduj odpowiedni� zak�adk�
#np. UNN- pr�dko�� d�wi�ku uzyskana za pomoc� NN
#U -pr�dko�� d�wi�ku literaturowa
tablece = xlsread (fullpath, 'UNN');
[rows,cols]=size(tablece);
rows=rows-1;
cols=cols-1;
T=tablece(1,2:cols+1);
P=tablece(2:rows+1,1);
P(1)=P(1)/1;
P(2:end)=P(2:end)*1;#TODO
ce=tablece(2:rows+1,2:cols+1);
[coefc{1},~,coefc{2}]=polyfit(T,ce(1,:),2);
cs=@(T) polyval(coefc{1},T,[],coefc{2});


%Oblicz alphaP
alphaPs=@(T) -(polyval(polyder(coefrho{1}),T,[],coefrho{2})/coefrho{2}(2))./rhos(T);
% Oblicz KappaT
kappaTs=@(T) (1./(cs(T).^2)+T.*alphaPs(T).^2./Cps(T))./rhos(T);
% Dopasowanie
[coeflogkT{1},~,coeflogkT{2}]=polyfit(T,log(kappaTs(T)),3);

% Wyg�ad� I dopasuj f. pr�dko�ci d�wi�ku
for j=1:cols
ind=find(ce(:,j)>0);
pc=polyfit(P(ind),ce(ind,j).^3,2);
cef(:,j)=(polyval(pc,P)).^(1/3);
end
% dane pocz�tkowe
rhoa(1,:)=rhos(T);
Cpf(1,:)=Cps(T);
alpha(1,:)=alphaPs(T);
pa=polyfit(T,alpha(1,:),2);
dalpha(1,:)=polyval(polyder(pa),T);

% Metoda Heuna rozwi�zywania ODE
for j=2:rows;
% ind=find(ce(j,:)>0);
ind=1:cols;
% Krok ci�nienia
h=(P(j)-P(j-1))*1e6;
% Pierwszy krok- predyktor
dCpf(j,ind)=-(T(ind)./rhoa(j-1,ind)).*(alpha(j-1,ind).^2+dalpha(j-1,ind));
Cpf(j,ind)=Cpf(j-1,ind)+dCpf(j,ind);
drhoa(j,ind)=(T(ind).*alpha(j-1,ind).^2./Cpf(j-1,ind)+1./(cef(j-1,ind).^2));
rhoa(j,ind)=rhoa(j-1,ind)+drhoa(j,ind).*h;
pr=polyfit(T(ind),rhoa(j,ind),2);
alpha(j,ind)=-polyval(polyder(pr),T(ind))./rhoa(j,ind);
pa=polyfit(T(ind),alpha(j,ind),1);
dalpha(j,ind)=polyval(polyder(pa),T(ind));
% Drugi krok: korektor
dCpfe(j,ind)=-(T(ind)./rhoa(j,ind)).*(alpha(j,ind).^2+dalpha(j,ind));
Cpf(j,ind)=Cpf(j-1,ind)+(dCpf(j,ind)+dCpfe(j,ind))*h/2;
drhoae(j,ind)=(T(ind).*alpha(j,ind).^2./Cpf(j,ind)+1./(cef(j,ind).^2));
rhoa(j,ind)=rhoa(j-1,ind)+(drhoa(j,ind)+drhoae(j,ind))*h/2;
pr=polyfit(T(ind),rhoa(j,ind),2);
alpha(j,ind)=-polyval(polyder(pr),T(ind))./rhoa(j,ind);
pa=polyfit(T(ind),alpha(j,ind),1);
dalpha(j,ind)=polyval(polyder(pa),T(ind));
end

% Wybierz nazw� pliku Excel
[filename, filepath] = uiputfile('*.xlsx', 'Podaj nazw� pliku Excel do zapisu');

if isequal(filename, 0)
    disp('Anulowano wyb�r pliku. Zapis do pliku Excel zosta� przerwany.');
else
    fullpath = fullfile(filepath, filename);

    % Zapisz tablice 'ce' na arkuszu 'U_exp'
    xlswrite(fullpath, ce, 'U_exp');

    % Zapisz tablice 'cef' na arkuszu 'U_acc'
    xlswrite(fullpath, cef, 'U_acc');

    % Zapisz tablice 'rhoe' na arkuszu 'RHO_exp'
    #xlswrite(fullpath, rhoe, 'RHO_exp');

    % Zapisz tablice 'rhoa' na arkuszu 'RHO_acc'
    xlswrite(fullpath, rhoa, 'RHO_acc');

    % Zapisz tablice 'Cpf' na arkuszu 'Cp'
    xlswrite(fullpath, Cpf, 'Cp');

    % Zapisz tablice 'alpha' na arkuszu 'Alpha_P'
    xlswrite(fullpath, alpha*1e4, 'Alpha_P');

    disp(['Dane zosta�y zapisane do pliku Excel: ' fullpath]);
end

#Stw�rz wykresy do wgl�du
figure
subplot(3,1,1)
plot(Td,d,'s')
hold on
#plot(T,rhoe(1,:),'o')
plot(Td,rhos(Td))
xlabel('T, K')
ylabel('\rho, kg/m^3')
set(gca,'FontSize',FS)
subplot(3,1,2)
plot(TCp,Cp,'s')
hold on
plot(T,Cps(T),'.-')
xlabel('T, K')
ylabel('C_p, J/(kg K)')
set(gca,'FontSize',FS)
subplot(3,1,3)
plot(T,ce(1,:),'s')
hold on
plot(T,cs(T),'.-')
xlabel('T, K')
ylabel('c, m/s')
set(gca,'FontSize',FS)
figure

xlabel('P, MPa')
ylabel('C_p, J/(kg K)')
set(gca,'FontSize',FS)
figure
plot(P,alpha,'*-')
xlabel('P, MPa')
ylabel('\alpha_P, kg/m^3')
set(gca,'FontSize',FS)
figure
plot(P,Cpf,'*-')
xlabel('P, MPa')
ylabel('C_P, kg/m^3')
t(gca,'FontSize',FS)