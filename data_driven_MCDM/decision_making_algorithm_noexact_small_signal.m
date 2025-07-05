close all; clear all; clc;
addpath("../Test_daily_profile/Intra_day_forecast_paper/");
% addpath("C:\Users\Francesca\Documents\gridofgridsML\Matlab_codes\Functions Optimitzacio\");
format long;

run create_list_of_CCRCs.m

selected_combinations=double(pyrunfile('CCRCs_selected.py','combinations_selected'));

list_indicators ={'H2_freq', 'H2_vdc','DCgain_freq','DCgain_vdc'};%

list_indicators_index = [2 3 4 5]; %  % index of the indicator in the excel file with exact small-signal stability results

% constr_num_changes=1;
% constr_num_changes_start=constr_num_changes-1;

weight_ind = [1 1 1 1] % weights of each indicator

CCRC_prev_OP=-1; % initialize the CCRC of previous timestamp

%% Power flows
X_PF = readmatrix('X_PF.xlsx');
X_IPC = readmatrix(['X_IPC.xlsx']);

X_PF_IPC=[X_PF,X_IPC];

%%
% Define column names
columnNames = {'Exec_time','CCRC','CCRC_index','num_changes','Obj_fun','H2_freq','H2_vdc','DCgain_freq','DCgain_vdc','N_of_checked_CCRCs','checked_CCRCs'};

% Initialize an empty table
T_results = table('Size', [0, length(columnNames)], ...
                   'VariableTypes', repmat({'double'}, 1, length(columnNames)), ...
                   'VariableNames', columnNames);
%%
deltas=zeros([96,1]);
decide_first_CCRC=0;
decided_comb=17; % Assum the system is operating with CCRC 17 that is the one stable in the majority of cases
%%
% == Main loop ==
for iisamples=1:96%length(Pload_tot_vect)
    constr_num_changes=1;
    iisamples
   
    tic();

    list_stable_CCRCs_at_OP=[];
    T_indicators = table();
    for ii_ind=1:length(list_indicators)
        str_indicator = string(list_indicators(ii_ind));
        T_indicators.(str_indicator)={[]};
    end

    for ii=1:length(selected_combinations)
        t_stab_file = readmatrix(['Stab_H2_DCgain_CCRC_',num2str(selected_combinations(ii)),'_daily_prof.xlsx']);
        stab = t_stab_file(iisamples,1);

        if stab
            list_stable_CCRCs_at_OP = [list_stable_CCRCs_at_OP, selected_combinations(ii)];
        end
    end

    if CCRC_prev_OP==-1
        CCRC_stables_constr=[];

        if decide_first_CCRC==1
            t_stab_file = readmatrix(['Stab_H2_DCgain_CCRC_',num2str(decided_comb),'_daily_prof.xlsx']);
            pred = t_stab_file(iisamples,list_indicators_index)'; % EXACT: get indicator from the excel file

            for ii_ind = 1:length(list_indicators)
                str_indicator = string(list_indicators(ii_ind)); % name of the indicator as string     
                T_indicators.(str_indicator) = {[T_indicators.(str_indicator){:} pred(ii_ind)]};
            end

            CCRC_stables_constr=[CCRC_stables_constr,decided_comb];
            T_decision = zeros(1,1); %--> Decide by min sum indicators
            for ii_ind=1:length(list_indicators)
                str_indicator = string(list_indicators(ii_ind));
                T_decision = T_decision + T_indicators.(str_indicator){:}.*weight_ind(ii_ind);
            end
        else
            for ii=1:length(list_stable_CCRCs_at_OP)
                t_stab_file = readmatrix(['Stab_H2_DCgain_CCRC_',num2str(list_stable_CCRCs_at_OP(ii)),'_daily_prof.xlsx']);
    
                pred = t_stab_file(iisamples,list_indicators_index)';%[t_stab_file(iisamples,5); t_stab_file(iisamples,6); t_stab_file(iisamples,7); t_stab_file(iisamples,10)];%  ,list_indicators_index(ii_ind)); % FAKE: get indicator from the excel file
    
                for ii_ind = 1:length(list_indicators)
                    str_indicator = string(list_indicators(ii_ind)); % name of the indicator as string     
                    T_indicators.(str_indicator) = {[T_indicators.(str_indicator){:} pred(ii_ind)]};
                end
                CCRC_stables_constr=[CCRC_stables_constr,list_stable_CCRCs_at_OP(ii)];
    
            end
    
            T_decision = zeros(1,length(list_stable_CCRCs_at_OP)); %--> Decide by min sum indicators
            for ii_ind=1:length(list_indicators)
                str_indicator = string(list_indicators(ii_ind));
                T_decision = T_decision + T_indicators.(str_indicator){:}.*weight_ind(ii_ind);
            end
        end
    else
        constr_num_changes_respected=0;
        CCRC_stables_constr=[];
        while constr_num_changes_respected==0 && constr_num_changes<=6
            for ii=1:length(list_stable_CCRCs_at_OP)
                
                if ismember(list_stable_CCRCs_at_OP(ii),list_of_stable_CCRCs_prev_OP)
                    num_changes = sum(table2array(T_combinacions_viables(list_stable_CCRCs_at_OP(ii),[1:6]))~=table2array(T_combinacions_viables(T_results.CCRC(iisamples-1),[1:6])));
                    if num_changes<=constr_num_changes %num_changes>= constr_num_changes-1 && 
                        constr_num_changes_respected=1;
                        t_stab_file = readmatrix(['Stab_Hinf_H2_comb',num2str(list_stable_CCRCs_at_OP(ii)),'_daily_prof.xlsx']);

                        pred = t_stab_file(iisamples,list_indicators_index)';
                        pred = double(py.array.array('d', py.numpy.nditer(pred)));

                        for ii_ind = 1:length(list_indicators)
                            str_indicator = string(list_indicators(ii_ind)); % name of the indicator as string     
                            T_indicators.(str_indicator) = {[T_indicators.(str_indicator){:} pred(ii_ind)]};

                        end
                        CCRC_stables_constr=[CCRC_stables_constr,list_stable_CCRCs_at_OP(ii)];
                    end
                end            
            end
            if constr_num_changes_respected==0
                constr_num_changes=constr_num_changes+1;
            end
        end
        
        T_decision = zeros(1,length(CCRC_stables_constr)); %--> Decide by min sum indicators
        for ii_ind=1:length(list_indicators)
            str_indicator = string(list_indicators(ii_ind));
            T_decision = T_indicators.(str_indicator){:}.*weight_ind(ii_ind)+T_decision;
        end
        T_decision =  T_decision - sum(table2array(T_results(iisamples-1,list_indicators_scaled).*weight_ind));

    end
    
    [min_val, min_val_index] = min(T_decision);
    
    deltas(iisamples,1)=T_indicators.H2_freq{1}(min_val_index)-min(T_indicators.H2_freq{1});
    deltas(iisamples,2)=T_indicators.H2_vdc{1}(min_val_index)-min(T_indicators.H2_vdc{1});
    deltas(iisamples,3)=T_indicators.DCgain_freq{1}(min_val_index)-min(T_indicators.DCgain_freq{1});
    deltas(iisamples,4)=T_indicators.DCgain_vdc{1}(min_val_index)-min(T_indicators.DCgain_vdc{1});

    % if T_indicators.H2_freq{1}(min_val_index)~=min(T_indicators.H2_freq{1})
    % stop=1;
    % elseif T_indicators.DCgain_freq{1}(min_val_index)~=min(T_indicators.DCgain_freq{1})
    % stop=1;
    % end
    
    T_results.Exec_time(iisamples)=toc;
    if CCRC_prev_OP==-1  
        if decide_first_CCRC
            T_results.CCRC(iisamples)=decided_comb;
        else
            T_results.CCRC(iisamples)=list_stable_CCRCs_at_OP(min_val_index);
        end
    else
        T_results.CCRC(iisamples)=CCRC_stables_constr(min_val_index);
    end
    T_results.CCRC_index(iisamples) = find(selected_combinations==T_results.CCRC(iisamples));
    if CCRC_prev_OP==-1
        T_results.num_changes(iisamples)=-1;
    else
        num_changes = sum(table2array(T_combinacions_viables(T_results.CCRC(iisamples),[1:6]))~=table2array(T_combinacions_viables(T_results.CCRC(iisamples-1),[1:6])));
        T_results.num_changes(iisamples)=num_changes;
    end
    T_results.Obj_fun(iisamples)=min_val;
    for ii_ind=1:length(list_indicators)
        str_indicator = string(list_indicators(ii_ind));
        T_results.(str_indicator)(iisamples)=T_indicators_not_scaled.(str_indicator){1}(min_val_index);%{[]};
    end   
    %T_results.list_of_stable_CCRCs(iisamples)=list_stable_CCRCs_at_OP;
    T_results.N_of_checked_CCRCs(iisamples)=length(CCRC_stables_constr);
    list_of_stable_CCRCs_prev_OP=list_stable_CCRCs_at_OP;
    CCRC_prev_OP=T_results.CCRC(iisamples);
    T_results.checked_CCRCs(iisamples)=string(num2str(CCRC_stables_constr));
end

% writetable(T_results,'./Decision_Making/ExactModels_noweigths_dailyprof_V2.xlsx');%
