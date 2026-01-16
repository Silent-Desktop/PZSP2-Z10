param n;
set NODES = 1..n;
set EDGES = {i in NODES, j in NODES: i<j};
set EDGE_FLOWS = {i in NODES, j in NODES: (i,j) in EDGES or (j,i) in EDGES};
set CONNECTIONS = {i in NODES, j in NODES: i<j};
set TRANSPONDERS;

param demand {CONNECTIONS} default 0;
param base_width;
param encrypted_width;
param encrypt_root in {NODES} default 1;
param cap {TRANSPONDERS};
param cost {TRANSPONDERS};

var usage {EDGE_FLOWS, CONNECTIONS} integer, >=0;
var choice {CONNECTIONS, TRANSPONDERS} integer, >=0;
var path_total {(k,l) in CONNECTIONS} = sum {j in NODES: (k,j) in EDGE_FLOWS} usage[k,j,k,l];
var encrypted {EDGES} binary;
var encrypt_flow {(i,j) in EDGE_FLOWS, k in NODES: k!=encrypt_root} binary;

minimize total_cost:
sum {(k,l) in CONNECTIONS, m in TRANSPONDERS} cost[m]*choice[k,l,m];

subject to choice_demand {(k,l) in CONNECTIONS}:
sum {m in TRANSPONDERS} choice[k,l,m]*cap[m]
	>= demand[k,l];
	
subject to choice_usage {(k,l) in CONNECTIONS}:
sum {m in TRANSPONDERS} choice[k,l,m]
	= path_total[k,l];

subject to usage_path {(k,l) in CONNECTIONS, i in NODES: i!=k and i!=l}:
sum {j in NODES: (i,j) in EDGE_FLOWS} usage[i,j,k,l]
	= sum {j in NODES: (j,i) in EDGE_FLOWS} usage[j,i,k,l];

subject to usage_dst {(k,l) in CONNECTIONS}:
sum {i in NODES: (i,l) in EDGE_FLOWS} usage[i,l,k,l]
	= path_total[k,l];

subject to limit_edges {(i,j) in EDGES}:
sum {(k,l) in CONNECTIONS} (usage[i,j,k,l]+usage[j,i,k,l])
	<= base_width + encrypted[i,j]*(encrypted_width-base_width);

subject to encrypt_flow_root {k in NODES: k!=encrypt_root}:
sum {j in NODES: (encrypt_root,j) in EDGE_FLOWS} encrypt_flow[encrypt_root,j,k]
	= 1 + sum {i in NODES: (i,encrypt_root) in EDGE_FLOWS} encrypt_flow[i,encrypt_root,k];

subject to encrypt_flow_path {i in NODES, k in NODES: i!=k and i!=encrypt_root and k!=encrypt_root}:
sum {j in NODES: (i,j) in EDGE_FLOWS} encrypt_flow[i,j,k]
	= sum {j in NODES: (j,i) in EDGE_FLOWS} encrypt_flow[j,i,k];

subject to encrypt_edges {(i,j) in EDGES, k in NODES: k!=encrypt_root}:
	encrypt_flow[i,j,k] + encrypt_flow[j,i,k] <= 2*encrypted[i,j];
