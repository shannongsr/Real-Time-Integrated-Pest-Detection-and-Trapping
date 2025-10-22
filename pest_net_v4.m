%% compare_four_algos_standard_only.m  (CBG version)
% 场景：用 3 列 × 2 行（共 6 个）斜四边形替代原 3×3；其余逻辑/参数保持不变
% 对比四法：Equal spacing / Random connected / Standard Greedy / Cover-Bridge Greedy (CBG)
% 窗口1：2×2 布局（同 N*）；窗口2：成本–覆盖率曲线（随机连通含25–75%分位带）
% N* 选取：四条曲线中最先达到 theta 的最小 N；若都未达，则取覆盖最大处的 N
% 字体：Times New Roman

clear; close all; clc; rng(2025);

% ---------- 全局绘图字体 ----------
set(groot,'DefaultAxesFontName','Times New Roman');
set(groot,'DefaultTextFontName','Times New Roman');
set(groot,'DefaultLegendFontName','Times New Roman');
set(groot,'DefaultAxesFontSize',11);
set(groot,'DefaultTextFontSize',11);
set(groot,'DefaultLegendFontSize',10);
set(groot,'DefaultFigureColor','w');

%% ================== 参数 ==================
r_w  = 15;                 % 工作半径
r_c  = 500;                % 连通半径
theta = 0.90;              % 目标覆盖阈值
grid_step   = 5;           % 采样网格
budget_list = 6:48;        % 设备数量预算扫描
cand_step   = 6;           % 候选点沿折线的步距
rand_trials = 30;          % 随机连通试验次数

% —— CBG 的额外权重（稳健默认）——
div_w = 0.15;              % 多样性（分散度）权重：越大越鼓励"彼此更远"的点
seed_ratio = 0.8;          % 阶段A的种子比例（先覆盖+分散，再缝合）

%% ================== 场景（已替换为 3×2，外轮廓保持长方形） ==================
t2.nRow=3; t2.nCol=3; t2.outerBund=6; t2.gapW=3.5;
t2.col_w_narrow=[10 20]; t2.col_w_normal=[30 60]; t2.narrow_cols=1+randi(2);
t2.row_h_narrow=[20 40]; t2.row_h_normal=[30 60]; t2.narrow_rows=1+randi(2);
geom = build_standard_3x3(t2);

U = sample_points_in_polys(geom.paddies, grid_step);
V = points_along_polylines_by_spacing(geom.polylines, cand_step);
V = unique(round(V,3),'rows');

C_UV = pdist2(U,V) <= r_w;     
DistVV = pdist2(V,V);
AdjVV  = DistVV <= r_c;  AdjVV(1:size(V,1)+1:end)=false;

%% ================== 扫描四法曲线，确定 N* ==================
cov_eq = zeros(size(budget_list));
cov_rd_mean = zeros(size(budget_list));
cov_rd_lo   = zeros(size(budget_list));
cov_rd_hi   = zeros(size(budget_list));
cov_gr = zeros(size(budget_list));
cov_cbg = zeros(size(budget_list));

layout_rd_median = containers.Map('KeyType','double','ValueType','any');

for k=1:numel(budget_list)
    B = budget_list(k);

    % 1) Equal spacing
    D_eq = points_along_polylines_by_count(geom.polylines, B);
    [ceq,~] = eval_cover_connect(U, D_eq, r_w, r_c);
    cov_eq(k) = ceq;

    % 2) Random connected
    cov_trials = zeros(rand_trials,1);
    D_trials = cell(rand_trials,1);
    for t=1:rand_trials
        D_rd = random_connected(V, B, AdjVV, DistVV);
        D_trials{t} = D_rd;
        [crd,~] = eval_cover_connect(U, D_rd, r_w, r_c);
        cov_trials(t) = crd;
    end
    cov_rd_mean(k) = mean(cov_trials);
    cov_rd_lo(k)   = prctile(cov_trials,25);
    cov_rd_hi(k)   = prctile(cov_trials,75);
    [~,med_idx] = min(abs(cov_trials - median(cov_trials)));
    layout_rd_median(B) = D_trials{med_idx};

    % 3) Standard Greedy (你原来的连通贪心)
    D_gr = greedy_cover_connect(V, U, B, C_UV, AdjVV, DistVV);
    [cgr,~] = eval_cover_connect(U, D_gr, r_w, r_c);
    cov_gr(k) = cgr;

    % 4) Cover-Bridge Greedy (新的改进)
    D_cbg = cover_bridge_greedy_budget(U,V,C_UV,AdjVV,DistVV,B,div_w,seed_ratio);
    [ccbg,~] = eval_cover_connect(U, D_cbg, r_w, r_c);
    cov_cbg(k) = ccbg;
end

covBest = max([cov_eq; cov_rd_mean; cov_gr; cov_cbg], [], 1);
id_ok = find(covBest >= theta, 1, 'first');
if isempty(id_ok), [~, id_ok] = max(covBest); end
N_star = budget_list(id_ok);

%% ================== 用 N* 生成四法布局 ==================
D_fixed = points_along_polylines_by_count(geom.polylines, N_star);
D_rand  = layout_rd_median(N_star);
D_greedy= greedy_cover_connect(V, U, N_star, C_UV, AdjVV, DistVV);
D_cbg   = cover_bridge_greedy_budget(U,V,C_UV,AdjVV,DistVV,N_star,div_w,seed_ratio);

[c_eq,conn_eq] = eval_cover_connect(U, D_fixed, r_w, r_c);
[c_rd,conn_rd] = eval_cover_connect(U, D_rand,  r_w, r_c);
[c_gr,conn_gr] = eval_cover_connect(U, D_greedy, r_w, r_c);
[c_cbg,conn_cbg] = eval_cover_connect(U, D_cbg, r_w, r_c);

%% ================== 窗口1：2×2 布局对比 ==================
fig1 = figure('Name','Standard 3×2 Four Algorithms (N* comparison)','Color','w');
tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

ax = nexttile(1);
draw_layout_panel(ax, geom, U, D_fixed, r_w, ...
    sprintf('(a) Equal spacing  N=%d, cov=%.1f%%, conn=%d', N_star, 100*c_eq, conn_eq));

ax = nexttile(2);
draw_layout_panel(ax, geom, U, D_rand, r_w, ...
    sprintf('(b) Random connected  N=%d, cov=%.1f%%, conn=%d', N_star, 100*c_rd, conn_rd));

ax = nexttile(3);
draw_layout_panel(ax, geom, U, D_greedy, r_w, ...
    sprintf('(c) GR  N=%d, cov=%.1f%%, conn=%d', N_star, 100*c_gr, conn_gr));

ax = nexttile(4);
draw_layout_panel(ax, geom, U, D_cbg, r_w, ...
    sprintf('(d) CBG  N=%d, cov=%.1f%%, conn=%d', N_star, 100*c_cbg, conn_cbg));

%% ================== 窗口2：成本–覆盖率曲线 ==================
fig2 = figure('Name','Cost Coverage','Color','w');
ax2 = axes; hold(ax2,'on');

x = budget_list(:)'; ylo = cov_rd_lo(:)'; yhi = cov_rd_hi(:)';
patch([x fliplr(x)], [ylo fliplr(yhi)], [0.85 0.85 0.95], ...
      'EdgeColor','none','FaceAlpha',0.6,'Parent',ax2);

p_eq  = plot(ax2,budget_list, cov_eq,      '-^','LineWidth',1.6);
p_rd  = plot(ax2,budget_list, cov_rd_mean, '-o','LineWidth',1.6);
p_gr  = plot(ax2,budget_list, cov_gr,      '-s','LineWidth',1.6);
p_cbg = plot(ax2,budget_list, cov_cbg,     '-d','LineWidth',1.6);

xline(ax2,N_star,'k--','LineWidth',1.6);

ylim(ax2,[0 1]); xlim(ax2,[min(budget_list) max(budget_list)]);
xlabel(ax2,'\bf Cost = number of devices (N)');
ylabel(ax2,'\bf Coverage ratio');
title(ax2,'\bf Cost Coverage');

legend([p_eq p_rd p_gr p_cbg], ...
       {'Equal spacing','Random connected mean','GR','CBG'}, ...
       'Location','southoutside','Orientation','horizontal','Box','off');

set(ax2,'XGrid','off','YGrid','off','LineWidth',1.2,'FontWeight','bold','Box','on','Layer','top');

%% ================== 终端摘要 ==================
fprintf('== New scene: 3 columns × 2 rows (6 skewed quads; outer is RECTANGLE) ==\n');
fprintf('Trade-off: theta=%.0f%%, N*=%d\n', 100*theta, N_star);
fprintf('  Equal spacing @N*: cov=%5.1f%%, conn=%d\n', 100*c_eq, conn_eq);
fprintf('  Random conn.  @N*: cov=%5.1f%%, conn=%d\n', 100*c_rd, conn_rd);
fprintf('  GR        @N*: cov=%5.1f%%, conn=%d\n', 100*c_gr, conn_gr);
fprintf('  CBG           @N*: cov=%5.1f%%, conn=%d\n', 100*c_cbg, conn_cbg);

%% ================== 函数区 ==================
function geom = build_standard_3x3(p)
% 3×2 布局；外轮廓严格长方形。
% 内部分隔：两条竖线允许小斜率，横向分隔为水平直线。
% ——保持与旧接口一致——

% --- 基本尺寸（保持与你当前脚本数量级接近） ---
Wi = 65;                 % 内域宽（不含 outerBund）
Hi = 85;                 % 内域高
b  = p.outerBund;        % 外围留白

% 列/行比例（中间列稍窄；上行稍窄）
col_ratio = [0.45, 0.25, 0.21]; col_ratio = col_ratio/sum(col_ratio);
row_ratio = [0.42, 0.58];      row_ratio = row_ratio/sum(row_ratio);

% 竖向分隔线的"斜率"
k1 = -0.14;
k2 = -0.12;

% --- 计算列/行边界（以 y=0 的下边为基准） ---
xc = [0, cumsum(col_ratio)*Wi];   % = [0, x1, x2, Wi]
yc = [0, cumsum(row_ratio)*Hi];   % = [0, yh, Hi]

% 定义 4 条"竖向边界线" Li(y) = a_i + s_i * y
a = [xc(1), xc(2), xc(3), xc(4)];
s = [0,     k1,    k2,    0   ];

% 构造 6 个四边形
rects = cell(6,1); idx=0;
for r = 1:2
    y0 = yc(r); y1 = yc(r+1);
    for c = 1:3
        xL0 = a(c)   + s(c)   * y0;
        xR0 = a(c+1) + s(c+1) * y0;
        xL1 = a(c)   + s(c)   * y1;
        xR1 = a(c+1) + s(c+1) * y1;
        idx = idx + 1;
        rects{idx} = [xL0 y0; xR0 y0; xR1 y1; xL1 y1];   % 顺时针
    end
end

% 外轮廓：严格长方形
Outer = [0 0; Wi 0; Wi Hi; 0 Hi];

% 中心折线：两条竖线 + 一条水平线（用于候选点/锚点）
v1 = [a(2)+s(2)*0, 0;     a(2)+s(2)*Hi, Hi];
v2 = [a(3)+s(3)*0, 0;     a(3)+s(3)*Hi, Hi];
h1 = [0, yc(2); Wi, yc(2)];

% 平移到正坐标并留白
allPts = Outer;
for i=1:6, allPts=[allPts; rects{i}]; end %#ok<AGROW>
shift = [b - min(allPts(:,1)), b - min(allPts(:,2))];
for i=1:6, rects{i} = rects{i} + shift; end
Outer = Outer + shift; v1=v1+shift; v2=v2+shift; h1=h1+shift;

% 输出
geom.paddies   = cellfun(@(P) polyshape(P), rects, 'UniformOutput', false);
geom.outerRect = polyshape(Outer);
geom.pathRects = {};
geom.polylines = { [Outer; Outer(1,:)], v1, v2, h1 };

geom.Wtot = max(Outer(:,1)) + b; 
geom.Htot = max(Outer(:,2)) + b;
end

function U = sample_points_in_polys(Pcells, step)
xmin=inf; xmax=-inf; ymin=inf; ymax=-inf;
for i=1:numel(Pcells)
    [x,y] = boundary(Pcells{i});
    xmin=min(xmin,min(x)); xmax=max(xmax,max(x));
    ymin=min(ymin,min(y)); ymax=max(ymax,max(y));
end
[xg,yg]=meshgrid(xmin:step:xmax, ymin:step:ymax);
X=xg(:); Y=yg(:);
mask=false(size(X));
for i=1:numel(Pcells)
    mask = mask | isinterior(Pcells{i}, X, Y);
end
U=[X(mask), Y(mask)];
end

function D = points_along_polylines_by_spacing(polylines, s)
D=[]; for i=1:numel(polylines)
    P=polylines{i}; seg=diff(P); seglen=vecnorm(seg,2,2); L=sum(seglen);
    if L<=0, continue; end
    n=max(1, round(L/s)); t=linspace(0,L,n+1)'; t(end)=[];
    D=[D; sample_along_polyline(P, seglen, t)]; %#ok<AGROW>
end
D=unique(round(D,3),'rows');
end

function D = points_along_polylines_by_count(polylines, N)
Llist=zeros(numel(polylines),1); Segs=cell(numel(polylines),1); Seglen=cell(numel(polylines),1);
for i=1:numel(polylines)
    P=polylines{i}; seg=diff(P); seglen=vecnorm(seg,2,2); Llist(i)=sum(seglen); Segs{i}=P; Seglen{i}=seglen; end
Lsum=sum(Llist); if Lsum==0, D=zeros(0,2); return; end
t=linspace(0,Lsum,N+1)'; t(end)=[]; D=zeros(N,2); cum=[0;cumsum(Llist)];
for k=1:N
    tk=t(k); idx=find(cum<=tk,1,'last'); if idx==numel(cum), idx=numel(cum)-1; end
    tk_local=tk - cum(idx); D(k,:)=sample_along_polyline(Segs{idx},Seglen{idx},tk_local);
end
D=unique(round(D,3),'rows');
end

function p = sample_along_polyline(P, seglen, t)
cum=[0;cumsum(seglen)];
if numel(t)>1, p=zeros(numel(t),2);
    for i=1:numel(t), ti=t(i); k=find(cum<=ti,1,'last'); if k==numel(seglen)+1, k=numel(seglen); end
        dt=(ti-cum(k))/seglen(k); p0=P(k,:); p1=P(k+1,:); p(i,:)=(1-dt)*p0+dt*p1; end
else, ti=t; k=find(cum<=ti,1,'last'); if k==numel(seglen)+1, k=numel(seglen); end
    dt=(ti-cum(k))/seglen(k); p0=P(k,:); p1=P(k+1,:); p=(1-dt)*p0+dt*p1; end
end

function [coverage, connected] = eval_cover_connect(U, D, r_w, r_c)
if isempty(D), coverage=0; connected=0; return; end
coverage = mean(min(pdist2(U,D),[],2) <= r_w);
Adj = pdist2(D,D) <= r_c; Adj(1:size(D,1)+1:end)=false;
visited=false(size(D,1),1); q=1;
while ~isempty(q)
    v=q(1); q(1)=[]; if ~visited(v), visited(v)=true; q=[q find(Adj(v,:))]; end
end
connected = all(visited);
end

function D = greedy_cover_connect(V, U, B, C_UV, AdjVV, DistVV)
Nu=size(U,1); Nv=size(V,1);
covered=false(Nu,1); chosen=false(Nv,1);
gain0=sum(C_UV,1); [~,i0]=max(gain0);
chosen(i0)=true; covered=covered|C_UV(:,i0);
while nnz(chosen)<B
    free=~chosen; cand=find(free & any(AdjVV(:,chosen),2));
    if isempty(cand)
        dist2S=min(DistVV(:,chosen),[],2); dist2S(chosen)=inf;
        [~,j]=min(dist2S); chosen(j)=true; covered=covered|C_UV(:,j); continue;
    end
    valid_rows=find(~covered(:));
    if isempty(valid_rows), gains=zeros(1,numel(cand));
    else, gains=sum(C_UV(valid_rows,cand),1); end
    if all(gains<=0)
        dist2S=min(DistVV(cand,chosen),[],2); [~,pos]=min(dist2S); j=cand(pos);
    else
        [~,pos]=max(gains); j=cand(pos); end
    chosen(j)=true; covered=covered|C_UV(:,j);
end
D=V(chosen,:);
end

function D = cover_bridge_greedy_budget(U,V,C_UV,AdjVV,DistVV,B,div_w,seed_ratio)
% -------- Cover-Bridge Greedy (CBG) --------
% 阶段A（种子）：忽略连通，按 "覆盖 + 分散度" 选 ceil(seed_ratio*B)
% 阶段B（桥接）：尽量用最少节点把多个连通分量缝成一个，优先同时接触≥2个分量的点
% 阶段C（补点）：在保持与已选集合连通的前提下，用"覆盖 + 分散度"补满直至 B
Nu=size(U,1); Nv=size(V,1);
covered=false(Nu,1); chosen=false(Nv,1);

% 预计算覆盖增益（动态基于 covered 更新）
gain = @(j,coveredMask) sum(C_UV(~coveredMask,j));

% -------- 阶段A：覆盖+分散度 种子选择 --------
B_A = max(1, min(B, ceil(seed_ratio * B)));
% 先选覆盖能力最强的起点
gain0=sum(C_UV,1); [~,i0]=max(gain0);
chosen(i0)=true; covered=covered|C_UV(:,i0);

while nnz(chosen)<B_A
    free = find(~chosen);
    if isempty(free), break; end
    % 覆盖增益
    g = arrayfun(@(j) gain(j,covered), free);
    % 分散度：到已选集合的最近距离（越远越好）
    dist2S = min(DistVV(free, chosen), [], 2);
    score = g + div_w * dist2S(:);
    [~,pos]=max(score);
    j = free(pos);
    chosen(j)=true; covered=covered|C_UV(:,j);
end

% -------- 阶段B：桥接多个连通分量 --------
while nnz(chosen)<B
    comps = components_from_chosen(chosen, AdjVV);
    if numel(comps)<=1
        break; % 已经连成一个分量
    end
    free = find(~chosen);
    if isempty(free), break; end
    % 对每个候选点，统计它能触达多少个已选分量
    touch_counts = zeros(numel(free),1);
    cov_gain     = zeros(numel(free),1);
    dist2S       = zeros(numel(free),1);
    for t=1:numel(free)
        j = free(t);
        touch = 0;
        for c=1:numel(comps)
            if any(AdjVV(j, comps{c}))
                touch = touch + 1;
            end
        end
        touch_counts(t)=touch;
        cov_gain(t)    = gain(j,covered);
        if any(chosen)
            dist2S(t) = min(DistVV(j, chosen));
        else
            dist2S(t) = 0;
        end
    end
    % 优先选择能同时连接≥2个分量的节点；若并列，用覆盖+分散度打分
    maxTouch = max(touch_counts);
    if maxTouch>=2
        mask = (touch_counts==maxTouch);
        sc = cov_gain + div_w * dist2S;
        sc(~mask) = -inf;
        [~,idx]=max(sc);
        j = free(idx);
        chosen(j)=true; covered=covered|C_UV(:,j);
    else
        % 找不到"一点连两分量"的候选，则选择能连到当前任一分量且得分最高的点
        mask = touch_counts>=1;
        if any(mask)
            sc = cov_gain + div_w * dist2S;
            sc(~mask) = -inf;
            [~,idx]=max(sc);
            j = free(idx);
            chosen(j)=true; covered=covered|C_UV(:,j);
        else
            % 实在没有可触达分量的点（极端情况），退化为全局覆盖+分散度
            sc = cov_gain + div_w * dist2S;
            [~,idx]=max(sc);
            j = free(idx);
            chosen(j)=true; covered=covered|C_UV(:,j);
        end
    end
    if nnz(chosen)>=B, break; end
end

% -------- 阶段C：保持连通的前提下补点 --------
while nnz(chosen)<B
    free = find(~chosen);
    cand = free(any(AdjVV(free, chosen),2)); % 仅考虑与现有集合连通的候选
    if isempty(cand)
        % 若没有连通候选，选最近的一个补上（防止卡死）
        dist2S=min(DistVV(free,chosen),[],2); [~,p]=min(dist2S);
        j=free(p); chosen(j)=true; covered=covered|C_UV(:,j); continue;
    end
    g = arrayfun(@(j) gain(j,covered), cand);
    d = min(DistVV(cand, chosen), [], 2);
    score = g + div_w * d;
    [~,pos]=max(score);
    j=cand(pos);
    chosen(j)=true; covered=covered|C_UV(:,j);
end

D=V(chosen,:);
end

function comps = components_from_chosen(chosen, AdjVV)
% 返回已选子图的连通分量（索引集合 cell 数组）
idxS = find(chosen);
if isempty(idxS), comps = {}; return; end
AdjSS = AdjVV(idxS, idxS);
N = numel(idxS);
vis=false(N,1);
comps = {};
for i=1:N
    if vis(i), continue; end
    q=i; vis(i)=true; comp=i;
    while ~isempty(q)
        v=q(1); q(1)=[];
        nbr = find(AdjSS(v,:));
        for u=nbr
            if ~vis(u), vis(u)=true; q=[q u]; comp=[comp u]; end %#ok<AGROW>
        end
    end
    comps{end+1} = idxS(comp); %#ok<AGROW>
end
end

function D = random_connected(V, B, AdjVV, DistVV)
Nv=size(V,1); if Nv==0 || B==0, D=zeros(0,2); return; end
chosen=false(Nv,1); i0=randi(Nv); chosen(i0)=true;
while nnz(chosen)<B
    free=~chosen; cand=find(free & any(AdjVV(:,chosen),2));
    if isempty(cand)
        dist2S=min(DistVV(:,chosen),[],2); dist2S(chosen)=inf;
        [~,j]=min(dist2S); chosen(j)=true; continue;
    end
    j=cand(randi(numel(cand))); chosen(j)=true;
end
D=V(chosen,:);
end

function draw_layout_panel(ax, geom, U, D, r_w, panel_title)
axes(ax); cla; hold on;
draw_base(geom,[0.96 0.98 0.96],[0.94 0.97 0.94],[0.83 0.94 0.83],[0.90 0.83 0.70]);
plot(U(:,1),U(:,2),'.','Color',[.75 .9 .75],'MarkerSize',6);
plot(D(:,1),D(:,2),'ro','MarkerFaceColor','r','MarkerSize',5.5);
draw_work_radii(D, r_w);
axis equal tight; box on;
title(panel_title,'FontWeight','bold');
set(ax,'LineWidth',1.2,'FontWeight','bold');
end

function draw_base(geom,col_bg,col_bund,col_paddy,col_path)
rectangle('Position',[0 0 geom.Wtot geom.Htot],'FaceColor',col_bg,'EdgeColor','k'); hold on;
plot(geom.outerRect,'FaceColor',col_bund,'EdgeColor','none'); hold on;
for i=1:numel(geom.paddies)
    plot(geom.paddies{i},'FaceColor',col_paddy,'EdgeColor',[0.6 0.85 0.6]); hold on;
end
end

function draw_work_radii(D, r)
th=linspace(0,2*pi,80);
for i=1:size(D,1)
    plot(D(i,1)+r*cos(th), D(i,2)+r*sin(th),'r:');
end
end
