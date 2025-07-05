T_combinacions = table(1,1,1,1,1,1,true,true,false,false,'VariableNames',{'IPCA',...
                'IPCB','IPCC','IPCD','IPCF','IPCE',...
                'GFmUnitsAC1','GFmUnitsAC2','VdcControlDC1','VdcControlDC2'});
 for i = 2:1:3^5
    T_combinacions = [T_combinacions;T_combinacions(end,:)];
    T_combinacions{i,5} = T_combinacions{i,5} + 1;
    %T_combinacions
    if (T_combinacions{i,5}) == 4
             T_combinacions{i,5} = 1;
             T_combinacions{i,4} = T_combinacions{i,4} + 1;
    end
    if (T_combinacions{i,4}) == 4
             T_combinacions{i,4} = 1;
             T_combinacions{i,3} = T_combinacions{i,3} + 1;
    end
    if (T_combinacions{i,3}) == 4
             T_combinacions{i,3} = 1;
             T_combinacions{i,2} = T_combinacions{i,2} + 1;
    end
    if (T_combinacions{i,2}) == 4
             T_combinacions{i,2} = 1;
             T_combinacions{i,1} = T_combinacions{i,1} + 1;
    end
    
    %Restriccions

    %Minim un GF en AC1
    if (T_combinacions{i,1} == 1) || (T_combinacions{i,2} == 1) || (T_combinacions{i,4} == 1)
        T_combinacions{i,7} = true;
    else
        T_combinacions{i,7} = false;
    end
    %Minim un GF en AC2
    if (T_combinacions{i,3} == 1) || (T_combinacions{i,5} == 1)
        T_combinacions{i,8} = true;
    else
        T_combinacions{i,8} = false;
    end
    %Minim un Vdc en DC 1
    if (T_combinacions{i,1} == 2) || (T_combinacions{i,2} == 2) || (T_combinacions{i,3} == 2)
        T_combinacions{i,9} = true;
    else
        T_combinacions{i,9} = false;
    end
    %Minim un Vdc en DC2
    if (T_combinacions{i,4} == 2) || (T_combinacions{i,5} == 2)
        T_combinacions{i,10} = true;
    else
        T_combinacions{i,10} = false;
    end
        
end
         
rows = ( (T_combinacions.VdcControlDC1==true)  & (T_combinacions.VdcControlDC2==true)) ;

T_combinacions_viables = T_combinacions(rows,:);
