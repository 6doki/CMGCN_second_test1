# from _typeshed import Self
import torch
import torch.nn.functional as F
import torch.nn as nn


class DelayAware_gcn_operation(nn.Module):
    def __init__(self, adj, in_dim, out_dim, num_vertices, patterns, activation='GLU'):
        """
        图卷积模块
        :param adj: 邻接图
        :param in_dim: 输入维度
        :param out_dim: 输出维度
        :param num_vertices: 节点数量
        :param activation: 激活方式 {'relu', 'GLU'}
        """
        super(DelayAware_gcn_operation, self).__init__()
        self.adj = adj          # (4N,4N) 4倍节点数
        self.in_dim = in_dim    # 64
        self.out_dim = out_dim  # 64
        self.num_vertices = num_vertices    # N
        self.activation = activation    # 'GLU'

        # 新增部分
        self.patterns = nn.Parameter(patterns, requires_grad=False)
        self.num_patterns = patterns.shape[0]
        self.pattern_len = patterns.shape[1]

        # 投影层:将输入特征映射到pattern相同的维度以便计算相似度
        self.self.query_proj = nn.Linear(in_dim, self.pattern_len)

        # 模式特征提取层：将匹配到的Pattern映射回输入特征维度
        self.pattern_conv = nn.Linear(self.pattern_len, in_dim)

        # 融合门控机制：决定保留多少原特征，注入多少模式特征
        self.fusion_gate = nn.Linear(in_dim * 2, out_dim)

        assert self.activation in {'GLU', 'relu'}
        if self.activation == 'GLU':
            self.FC = nn.Linear(self.in_dim, 2 * self.out_dim, bias=True)   # (in=64, out= 64*2)
        else:
            self.FC = nn.Linear(self.in_dim, self.out_dim, bias=True)

    def forward(self, x, mask=None):

        # 模式匹配
        x_tmp = x.permute(1, 0, 2)  # 调整维度以便计算: (N, B, C) -> (B, N, C)

        # 将输入特征投影为 Query: (B, N, Pattern_Len)
        query = self.query_proj(x_tmp)

        # 计算与所有 Patterns 的相似度 (点积)
        # patterns: (K, Pattern_Len) -> 转置 (Pattern_Len, K)
        # score: (B, N, K) 表示每个节点当前时刻与 K 个模式的匹配程度
        score = torch.matmul(query, self.patterns.t())
        attn = torch.softmax(score, dim=-1)  # 归一化权重

        # 加权组合最匹配的 Patterns
        # (B, N, K) * (K, Pattern_Len) -> (B, N, Pattern_Len)
        matched_pattern_feat = torch.matmul(attn, self.patterns)

        # 映射回特征维度: (B, N, In_Dim)
        delay_feat = self.pattern_conv(matched_pattern_feat)

        # --- 过程 2: 特征融合 ---
        # 拼接原特征和延迟特征
        combined = torch.cat([x_tmp, delay_feat], dim=-1)
        # 计算门控值 (0~1)
        z = torch.sigmoid(self.fusion_gate(combined))
        # 融合：原特征 * z + 延迟特征 * (1-z)
        x_enhanced = x_tmp * z + delay_feat * (1 - z)

        # 转回原维度 (N, B, C) 以便进行后续的 GCN 操作
        x = x_enhanced.permute(1, 0, 2)

        adj = self.adj
        '''如果提供了mask，将邻接矩阵adj移动到与mask相同的设备上，并乘以mask，将无效节点对应的邻接关系置0，从而忽略掉那些不应考虑的边'''
        if mask is not None:
            adj = adj.to(mask.device) * mask

        x = torch.einsum('nm, mbc->nbc', adj.to(x.device), x)  # 4*N, B, Cin  对邻接矩阵adj(维度nm)和特征矩阵x(维度mbc)进行乘法操作

        if self.activation == 'GLU':
            lhs_rhs = self.FC(x)  # 全连接层输出：(4*N, B, 2*Cout)
            lhs, rhs = torch.split(lhs_rhs, self.out_dim, dim=-1)  # 拆分之后的lhs和rhs:(4*N, B, Cout)
            out = lhs * torch.sigmoid(rhs)
            del lhs, rhs, lhs_rhs
            return out
        elif self.activation == 'relu':
            return torch.relu(self.FC(x))  # 3*N, B, Cout


class STSGCM(nn.Module):
    def __init__(self, adj, in_dim, out_dims, num_of_vertices, activation='GLU'):
        """
        :param adj: 邻接矩阵
        :param in_dim: 输入维度
        :param out_dims: list 各个图卷积的输出维度
        :param num_of_vertices: 节点数量
        :param activation: 激活方式 {'relu', 'GLU'}
        """
        super(STSGCM, self).__init__()
        self.adj = adj  # (4N,4N)
        self.in_dim = in_dim    # 64
        self.out_dims = out_dims    # out_dims：[64,64,64]
        self.num_of_vertices = num_of_vertices  # N
        self.activation = activation
        self.gcn_operations = nn.ModuleList()  # 先初始化一个模块列表，用于存储多个gcn_operation

        '''第一个gcn_operation'''
        self.gcn_operations.append(   # 第0个
            gcn_operation(
                adj=self.adj,
                in_dim=self.in_dim,
                out_dim=self.out_dims[0],  # out_dims[0]：64
                num_vertices=self.num_of_vertices,
                activation=self.activation
            )
        )
        '''剩下的gcn_operations'''
        for i in range(1, len(self.out_dims)):  # 第1-2个
            self.gcn_operations.append(
                gcn_operation(
                    adj=self.adj,
                    in_dim=self.out_dims[i-1],  # 输入：前一个图卷积操作的输出维度
                    out_dim=self.out_dims[i],   # 输出：64
                    num_vertices=self.num_of_vertices,
                    activation=self.activation
                )
            )

    def forward(self, x, mask=None):
        """
        :param x: (3N, B, Cin)
        :param mask: (3N, 3N)
        :return: (N, B, Cout)
        """
        need_concat = []  # 空列表，用于存储每个gcn_operation的输出

        for i in range(len(self.out_dims)):
            x = self.gcn_operations[i](x, mask)     # 4N, B, Cin
            need_concat.append(x)

        # shape of each element is (1, N, B, Cout)
        '''计算每个gcn_operation的输出，取出第num_of_vertices到2*num_of_vertices个节点的输出，并在第一个维度上增加一个维度，结果维度(1,N,B,Cout)'''
        need_concat = [
            torch.unsqueeze(
                h[self.num_of_vertices: 2 * self.num_of_vertices], dim=0
            ) for h in need_concat
        ]

        '''将上面的输出在第一个维度上拼接'''
        out = torch.max(torch.cat(need_concat, dim=0), dim=0).values  # (N, B, Cout)
        del need_concat
        return out


class STSGCL(nn.Module):
    def __init__(self,
                 adj,
                 history,
                 num_of_vertices,
                 in_dim,
                 out_dims,
                 strides=4,
                 activation='GLU',
                 # temporal_emb=False,
                 temporal_emb=True,
                 # spatial_emb=False,
                 spatial_emb=True
                 ):
        """
        :param adj: 邻接矩阵
        :param history: 输入时间步长
        :param in_dim: 输入维度
        :param out_dims: list 各个图卷积的输出维度
        :param strides: 滑动窗口步长，local时空图使用几个时间步构建的，默认为4
        :param num_of_vertices: 节点数量
        :param activation: 激活方式 {'relu', 'GLU'}
        :param temporal_emb: 加入时间位置嵌入向量
        :param spatial_emb: 加入空间位置嵌入向量
        """
        super(STSGCL, self).__init__()
        print("-------------------进入STSGCL---------------------")
        self.adj = adj  # (4N,4N)
        self.strides = strides  # 4
        self.history = history  # 12
        self.in_dim = in_dim    # 64
        self.out_dims = out_dims    # [64,64,64]
        self.num_of_vertices = num_of_vertices  # N
        self.activation = activation    # 'GLU'
        self.temporal_emb = temporal_emb    # True
        self.spatial_emb = spatial_emb  # True
        self.conv1 = nn.Conv2d(self.in_dim, self.out_dims[-1], kernel_size=(1, 2), stride=(1, 1), dilation=(1, 3))  # Conv1d(64, 64, kernel_size=(1, 2), stride=(1, 1), dilation=(1, 3))
        self.conv2 = nn.Conv2d(self.in_dim, self.out_dims[-1], kernel_size=(1, 2), stride=(1, 1), dilation=(1, 3))  # Conv1d(64, 64, kernel_size=(1, 2), stride=(1, 1), dilation=(1, 3))
        # self.conv1 = nn.Conv2d(self.in_dim, self.out_dims[-1], kernel_size=(1, 2), stride=(1, 1), dilation=(1, 1))
        # self.conv2 = nn.Conv2d(self.in_dim, self.out_dims[-1], kernel_size=(1, 2), stride=(1, 1), dilation=(1, 1))

        self.STSGCMS = nn.ModuleList()
        for i in range(self.history - self.strides + 1):    # (0,...,8)
            self.STSGCMS.append(
                STSGCM(
                    adj=self.adj,
                    in_dim=self.in_dim,
                    out_dims=self.out_dims,
                    num_of_vertices=self.num_of_vertices,
                    activation=self.activation
                )
            )


        if self.temporal_emb:
            self.temporal_embedding = nn.Parameter(torch.FloatTensor(1, self.history, 1, self.in_dim))  # (1,12,1,64)
            # 1, T=12, 1, Cin=64

        if self.spatial_emb:
            self.spatial_embedding = nn.Parameter(torch.FloatTensor(1, 1, self.num_of_vertices, self.in_dim))   # (1,1,N,64)
            # 1, 1, N, Cin=64

        self.reset()  # 初始化嵌入向量的权重

    def reset(self):
        print("--------------进入STSGCL中的reset()-----------------")
        if self.temporal_emb:
            nn.init.xavier_normal_(self.temporal_embedding, gain=0.0003)

        if self.spatial_emb:
            nn.init.xavier_normal_(self.spatial_embedding, gain=0.0003)

    def forward(self, x, mask=None):
        """
        :param x: B, T, N, Cin    (x数据：表示有B个批次，每个批次有T个时间步，每个时间步有N个节点，输入特征维度是Cin)
        :param mask: (N, N)
        :return: B, T-3, N, Cout  (输出out数据:表示有B个批次，每个批次有T-3个时间步，每个时间步有N个节点，输出特征维度是Cout)
        """
        # 消融
        if self.temporal_emb:   # x: B=64, T=12, N, Cin=64      temporal_embedding:1, history=12, 1, Cin=64
            x = x + self.temporal_embedding     # x: B=64, T=12, N, Cin=64

        if self.spatial_emb:    # 1, 1, N, Cin=64   spatial_embedding: 1, 1, N , Cin=64
            x = x + self.spatial_embedding      # x: B=64, T=12, N, Cin=64

        #############################################
        # shape is (B, C, N, T)
        '''
        下面代码对应论文中：两个二维的扩张卷积神经网络（捕获全局的）
        '''
        data_temp = x.permute(0, 3, 2, 1)                 # 交换位置x(B=64, Cin=64, N, T=12)
        data_left = torch.sigmoid(self.conv1(data_temp))  # (64, 64, 358, 9)
        data_right = torch.tanh(self.conv2(data_temp))    # (64, 64, 358, 9)
        data_time_axis = data_left * data_right           # (64, 64, 358, 9)
        data_res = data_time_axis.permute(0, 3, 2, 1)     # (64,9,358,64) 再次交换位置:(B,T-3,N,Cin)
        # shape is (B, T-3, N, C)
        #############################################

        need_concat = []
        batch_size = x.shape[0]  # 64

        for i in range(self.history - self.strides + 1):
            t = x[:, i: i+self.strides, :, :]  # 从x中提取一个滑动窗口:t(B, self.stride, N, Cin),这里stride=4
            t = torch.reshape(t, shape=[batch_size, self.strides * self.num_of_vertices, self.in_dim])   #将t重塑为(B, self.stride*N, Cin)
            # (B, 4*N, Cin)
            t = self.STSGCMS[i](t.permute(1, 0, 2), mask)  # (4*N, B, Cin) -> (N, B, Cout)
            t = torch.unsqueeze(t.permute(1, 0, 2), dim=1)  # (N, B, Cout) -> (B, N, Cout) ->(B, 1, N, Cout)
            need_concat.append(t)

        mid_out = torch.cat(need_concat, dim=1)  # (B, T-3, N, Cout)

        # #打印 mid_out 和 data_res 的维度
        # print(f"mid_out shape: {mid_out.shape}")
        # print(f"data_res shape: {data_res.shape}")
        #
        # #确保 mid_out 和 data_res 的时间维度（T-3）相同
        # if mid_out.shape[1] != data_res.shape[1]:
        #     print("Warning: Time dimensions do not match!")
        #
        # # 如果 mid_out 的时间维度小于 data_res 的时间维度，可以对 data_res 进行裁剪
        # if mid_out.shape[1] < data_res.shape[1]:
        #     data_res = data_res[:, :mid_out.shape[1], :, :]  # 裁剪 data_res 的时间维度
        #
        # # 如果 data_res 的时间维度小于 mid_out 的时间维度，可以对 mid_out 进行裁剪
        # elif mid_out.shape[1] > data_res.shape[1]:
        #     mid_out = mid_out[:, :data_res.shape[1], :, :]  # 裁剪 mid_out 的时间维度

        out = mid_out + data_res
        del need_concat, batch_size
        return out


class output_layer(nn.Module):
    def __init__(self, num_of_vertices, history, in_dim, out_dim, hidden_dim=128, horizon=24):
        """
        预测层，注意在作者的实验中是对每一个预测时间step做处理的，也即他会令horizon=1
        :param num_of_vertices:节点数量
        :param history:输入时间步长
        :param in_dim: 输入维度
        :param hidden_dim:中间层维度
        :param horizon:输出(预测)时间步长
        """
        super(output_layer, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.history = history  # 3
        self.in_dim = in_dim    # 64
        self.out_dim = out_dim  #1
        self.hidden_dim = hidden_dim    # 128
        self.horizon = horizon  # 1

        '''
        FC1:第一个全连接层，将输入维度转换为隐藏维度
        FC2:第二个全连接层，将隐藏维度转换为输出维度
        '''
        self.FC1 = nn.Linear(self.in_dim * self.history, self.hidden_dim, bias=True)    # Linear(in_features=192, out_features=128, bias=True)
        #self.FC2 = nn.Linear(self.hidden_dim, self.horizon , bias=True)
        self.FC2 = nn.Linear(self.hidden_dim, self.horizon * self.out_dim, bias=True)   # 128, 1

    def forward(self, x):
        """
        :param x: (B, Tin, N, Cin)：（批次大小，输入时间步长，节点数，输入特征维度）
        :return: (B, Tout, N)：（批次大小，预测时间步长，节点数量）
        """
        batch_size = x.shape[0]  # B
        x = x.permute(0, 2, 1, 3)  # (B, Tin, N, Cin)-->(B, N, Tin, Cin)
        out1 = torch.relu(self.FC1(x.reshape(batch_size, self.num_of_vertices, -1)))  # (B, N, Tin, Cin) -> (B, N, Tin * Cin) -> (B, N, hidden128)
        out2 = self.FC2(out1)  # (B, N, hidden) -> (B, N, horizon*1) 这里的1是Cout也就是out_dim
        out2 = out2.reshape(batch_size, self.num_of_vertices, self.horizon, self.out_dim)  # (B, N, horizon, Cout) : (B, N, 1, 1)
        del out1, batch_size
        return out2.permute(0, 2, 1, 3)  # B, horizon, N, Cout
        # return out2.permute(0, 2, 1)  # B, horizon, N


class STFGNN(nn.Module):
    print("-------------------------进入STFGNN类----------------------------")
    def __init__(self, config, data_feature):  # config:模型配置参数  data_feature:数据的特征信息
        """

        :param adj: local时空间矩阵
        :param history: 输入时间步长
        :param num_of_vertices: 节点数量
        :param in_dim:输入维度
        :param hidden_dims: lists, 中间各STSGCL层的卷积操作维度
        :param first_layer_embedding_size: 第一层输入层的维度
        :param out_layer_dim: 输出模块中间层维度
        :param activation: 激活函数 {relu, GlU}
        :param use_mask: 是否使用mask矩阵对adj进行优化
        :param temporal_emb:是否使用时间嵌入向量
        :param spatial_emb:是否使用空间嵌入向量
        :param horizon:预测时间步长
        :param strides:滑动窗口步长，local时空图使用几个时间步构建的，默认为4
        """

        '''
        1，super(STFGNN, self):STFGNN是子类的名字，self是当前对象的引用。这样组合告诉python要在STFGNN的上下文中，调用它的父类的方法
        2，__init__():这是父类nn.Module的构造函数,调用之后来初始化父类中的所有成员变量和属性
        '''
        super(STFGNN, self).__init__()

        # self在”=“左侧:表示正在定义或者修改类实例的属性
        self.config = config
        self.data_feature = data_feature
        self.scaler = data_feature["scaler"]    # std: Z-score 标准化
        self.num_batches = data_feature["num_batches"]  # 批次数量

        # self在”=“右侧:表示正在访问类实例的属性或者方法，获取其值或调用其功能
        adj = self.data_feature["adj_mx"]  # 融合的邻接矩阵
        history = self.config.get("window", 12)     # 12 历史时间步长 决定模型每次输入多少历史数据进行预测
        num_of_vertices = self.config.get("num_nodes", None)  # 节点数量
        in_dim = self.config.get("input_dim", 1)    # 1
        out_dim = self.config.get("output_dim", 1)  # 1
        hidden_dims = self.config.get("hidden_dims", None)  # [[64, 64, 64], [64, 64, 64], [64, 64, 64]] 隐藏层
        first_layer_embedding_size = self.config.get("first_layer_embedding_size", None)    # 第一层全连接层的输出维度：64
        out_layer_dim = self.config.get("out_layer_dim", None)  # 128 输出层中间层维度
        activation = self.config.get("activation", "GLU")
        use_mask = self.config.get("mask")  # False
        temporal_emb = self.config.get("temporal_emb", True)    # True 使用时间嵌入
        spatial_emb = self.config.get("spatial_emb", True)  # True 使用空间嵌入
        horizon = self.config.get("horizon", 24)    # 12 输出的预测时间步长
        strides = self.config.get("strides", 4)  # 4 滑动窗口步长 用于控制每次处理的时间步数量

        # 将局部变量存储为实例属性
        self.adj = adj  # (4N，4N）
        self.num_of_vertices = num_of_vertices  # N
        self.hidden_dims = hidden_dims  # [[64, 64, 64], [64, 64, 64], [64, 64, 64]]
        self.out_layer_dim = out_layer_dim  # 128
        self.activation = activation    # "GLU"
        self.use_mask = use_mask    # false

        self.temporal_emb = temporal_emb    # true
        self.spatial_emb = spatial_emb  # true
        self.horizon = horizon  # 12
        self.strides = 4
        # self.strides = 3

        # 定义一个线性层，并存储为实例属性
        self.First_FC = nn.Linear(in_dim, first_layer_embedding_size, bias=True)        # in_dim=1, first_layer_embedding_size=64   1-->64
        self.STSGCLS = nn.ModuleList()
        self.STSGCLS.append(
            STSGCL(
                adj=self.adj, # 融合的图
                history=history,
                num_of_vertices=self.num_of_vertices,
                in_dim=first_layer_embedding_size,
                out_dims=self.hidden_dims[0], # 64
                strides=self.strides,
                activation=self.activation,
                temporal_emb=self.temporal_emb,
                spatial_emb=self.spatial_emb
            )
        )

        in_dim = self.hidden_dims[0][-1]    # 64 设置为第一层隐藏维度的最后一个值
        history -= (self.strides - 1)   # 12-3=9 更新history
        print("***********************")
        print("经过第一个STSGCLS后的历史时间步长history更新为：", history)

        for idx, hidden_list in enumerate(self.hidden_dims):    # hidden_dims: [[64, 64, 64], [64, 64, 64], [64, 64, 64]]
            print("idx值：", idx)
            if idx == 0:
                continue
            self.STSGCLS.append(    # 2个STSGCL
                STSGCL(
                    adj=self.adj,
                    history=history,    # 9,6
                    num_of_vertices=self.num_of_vertices,
                    in_dim=in_dim,  # 64
                    out_dims=hidden_list,
                    strides=self.strides,
                    activation=self.activation,
                    temporal_emb=self.temporal_emb,
                    spatial_emb=self.spatial_emb
                )
            )
            history -= (self.strides - 1)   # 9->6, 6->3 更新history
            print("***********************")
            print("经过第%s个STSGCL层后的历史时间步长history更新为%s：" %(idx+1,history))
            in_dim = hidden_list[-1]    # 64

        # predictLayer包含多个output_layer，每个output_layer负责在每个时间步长上生成预测。预测层是最后的输出层，用来将时序数据映射为具体的预测值。
        self.predictLayer = nn.ModuleList()
        for t in range(self.horizon):   #12
            self.predictLayer.append(
                output_layer(     # 多个output_layer，用于每个时间步长的预测
                    num_of_vertices=self.num_of_vertices,
                    history=history,    # 3
                    in_dim=in_dim,  # 64
                    out_dim=out_dim,  # 1
                    hidden_dim=out_layer_dim,   # 128
                    horizon=1
                )
            )

        if self.use_mask:
            mask = torch.zeros_like(self.adj)
            mask[self.adj != 0] = self.adj[self.adj != 0]
            self.mask = nn.Parameter(mask)
        else:   # False
            self.mask = None

    def forward(self, x):
        """
        :param x: (B, Tin, N, Cin) B:batch  Tin:history   N:num_of_vertices   Cin:in_dim
        :return: (B, Tout, N)
        """
        x = torch.relu(self.First_FC(x))  # B=64, Tin=12, N, Cin=1 -> (64,12,N, Cout=64)、
        # print("经过定义的First_FC层之后x的数据形状：",x.size())  # pems08_30:([64,12,170,64]) pems08_10:([64,12,170,64])
        for model in self.STSGCLS:
            x = model(x, self.mask)
        need_concat = []
        for i in range(self.horizon):   # 12 对每一个时间步长
            out_step = self.predictLayer[i](x)  # (B, 1, N, 1) 每个时间步的预测
            need_concat.append(out_step)
        out = torch.cat(need_concat, dim=1)  # B, Tout, N, 1 合并所有时间步的预测进行输出
        del need_concat
        return out

    print("-------------------------退出STFGNN类----------------------------")


