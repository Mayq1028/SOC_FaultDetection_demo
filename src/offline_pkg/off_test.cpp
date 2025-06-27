    #include <iostream>
    #include <vector>
    #include <fstream>
    #include <sstream>
    #include <random>
    #include <Eigen/Dense>
    #include <Eigen/QR>
    #include <algorithm>
    #include <iomanip>
    #include <fcntl.h>
    #include <unistd.h>
    #include <termios.h>
    #include <sys/ioctl.h>
    #include <chrono>
    #include <thread>
    #include <cstring>


    using namespace Eigen;
    using namespace std;


    // 串口初始化函数
    int init_serial_port(const char* port, int baudrate = B115200) {
        int fd = open(port, O_RDWR | O_NOCTTY);
        if (fd < 0) {
            return -1;
        }

        struct termios tty;
        if(tcgetattr(fd, &tty) != 0) {
            close(fd);
            return -1;
        }

        cfsetospeed(&tty, baudrate);
        cfsetispeed(&tty, baudrate);

        tty.c_cflag &= ~PARENB; // 无奇偶校验
        tty.c_cflag &= ~CSTOPB; // 1位停止位
        tty.c_cflag &= ~CSIZE;
        tty.c_cflag |= CS8;     // 8位数据位
        tty.c_cflag &= ~CRTSCTS; // 无硬件流控
        tty.c_cflag |= CREAD | CLOCAL; // 启用接收

        tty.c_lflag &= ~ICANON; // 非规范模式
        tty.c_lflag &= ~ECHO;   // 禁用回显
        tty.c_lflag &= ~ECHOE;
        tty.c_lflag &= ~ISIG;

        tty.c_iflag &= ~(IXON | IXOFF | IXANY); // 无软件流控
        tty.c_iflag &= ~(IGNBRK|BRKINT|PARMRK|ISTRIP|INLCR|IGNCR|ICRNL);

        tty.c_oflag &= ~OPOST; // 原始输出
        tty.c_oflag &= ~ONLCR;

        tty.c_cc[VMIN] = 1;    // 读取至少1个字符
        tty.c_cc[VTIME] = 5;   // 0.5秒超时

        if(tcsetattr(fd, TCSANOW, &tty) != 0) {
            close(fd);
            return -1;
        }

        return fd;
    }

    // 解析串口数据 (根据Python代码的协议格式)
    bool parse_serial_data(const string& data, double& voltage, double& current, int& k) {
        // 通道1数据: <ch1>:3998
        if (data.find("<ch1>:") == 0) {
            try {
                string value_str = data.substr(6);
                int value = stoi(value_str);
                voltage = value / 1000.0; // 毫伏转伏特
                current = 0.0; // 只有电压值
                cout << "开路电压：" << fixed << setprecision(3) << voltage << "V" << endl;
                return true;
            } catch (...) {
                return false;
            }
        }
        // 全通道数据: <all>:3998,4003,0
        else if (data.find("<all>:") == 0) {
            try {
                string values_part = data.substr(6);
                vector<string> values;
                stringstream ss(values_part);
                string token;
                
                while (getline(ss, token, ',')) {
                    values.push_back(token);
                }
                
                if (values.size() >= 3) {
                    double U = stoi(values[0]) / 1000.0;  // 电池端电压(V)
                    double U_p = stoi(values[1]) / 1000.0; // 极性电压(V)
                    double Rp = U;
                    double I_k = U_p / Rp;  // 计算电流(A)
                    
                    voltage = U;
                    current = I_k;
                    
                    cout << "U(" << k << "): " << fixed << setprecision(3) << U << "V, "
                        << "I(" << k << "): " << I_k << "A" << endl;
                    
                    k++; // 增加计数器
                    return true;
                }
            } catch (...) {
                return false;
            }
        }
        return false;
    }

    // 从串口读取一行数据
    string read_serial_line(int fd) {
        const int buf_size = 256;
        char buf[buf_size];
        string line;
        ssize_t n;
        
        while ((n = read(fd, buf, buf_size - 1)) > 0) {
            buf[n] = '\0';
            line += buf;
            
            // 检查是否收到换行符
            if (line.find('\n') != string::npos) {
                // 去掉换行符和可能的回车符
                line.erase(remove(line.begin(), line.end(), '\n'), line.end());
                line.erase(remove(line.begin(), line.end(), '\r'), line.end());
                return line;
            }
        }
        
        return "";
    }


    // 从文件读取向量数据
    VectorXd load_data(const string& filename) {
        ifstream file(filename);
        if (!file.is_open()) {
            cerr << "错误：无法打开文件 " << filename << endl;
            exit(1);
        }
        
        vector<double> data;
        string line;
        while (getline(file, line)) {
            if (!line.empty()) {
                stringstream ss(line);
                double value;
                while (ss >> value) {
                    data.push_back(value);
                }
            }
        }
        file.close();
        
        VectorXd vec(data.size());
        for (int i = 0; i < data.size(); i++) {
            vec(i) = data[i];
        }
        return vec;
    }

    // 计算状态空间矩阵
    void calculate_para_matrix(double R0, double Rp, double Cp, double Qn, double deltaT, double eta,
                Matrix2d& A, Vector2d& B, double& D) {
        double exp_term = exp(-deltaT/(Rp*Cp));
        A << exp_term, 0, 0, 1;
        B << Rp*(1 - exp_term), -eta*deltaT/Qn;
        D = -R0;
    }

    // 状态生成函数
    void generate_state(const Matrix2d& A, const Vector2d& B, double D, const VectorXd& Cur,
                const Vector2d& x1, const VectorXd& pfit, double ww, double vv,
                MatrixXd& x, VectorXd& y) {
        int N = Cur.size();
        x.resize(2, N);
        y.resize(N-1);
        x.col(0) = x1;

        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis_w(-ww, ww);
        uniform_real_distribution<> dis_v(-vv, vv);

        for (int k = 0; k < N-1; k++) {
            Vector2d w(dis_w(gen), dis_w(gen));
            double v = dis_v(gen);
            x.col(k+1) = A * x.col(k) + B * Cur(k) + w;
            
            // 多项式求值
            double OCV = 0.0;
            for (int i = 0; i < pfit.size(); i++) {
                OCV += pfit[i] * pow(x(1, k), pfit.size()-1-i);
            }
            y(k) = OCV - x(0, k) + D * Cur(k) + v;
        }
    }

    // 降阶函数
    MatrixXd jiangjie(const MatrixXd& G, int s) {
        int n_z = G.rows(); // 获取行数
        int r = G.cols();   // 获取列数
        
        // 步骤1: 计算各列的欧氏范数并排序
        VectorXd Gn(r);
        for (int j = 0; j < r; ++j) {
            Gn(j) = G.col(j).norm(); // 计算欧氏范数
        }

        // 创建索引并降序排序
        std::vector<int> indices(r);
        for (int i = 0; i < r; ++i) indices[i] = i;
        std::sort(indices.begin(), indices.end(), [&](int a, int b) {
            return Gn(a) > Gn(b);
        });

        // 按新顺序排列矩阵
        MatrixXd Gx0(n_z, r);
        for (int j = 0; j < r; ++j) {
            Gx0.col(j) = G.col(indices[j]);
        }

        // 情况1: r <= s 直接返回
        if (r <= s) {
            return Gx0;
        }
        // 情况2: r > s
        else {
            // 错误检查
            if (s < n_z) {
                throw std::invalid_argument("error! s must be >= n_z");
            }
            
            int k = s - n_z; // 计算要保留的列数
            
            // 分割矩阵为两部分
            MatrixXd Gx01 = Gx0.leftCols(k);            // 前k列
            MatrixXd Gx02 = Gx0.rightCols(r - k);        // 剩余列
            
            // 计算Q矩阵 (对角线为行绝对值和)
            MatrixXd Q = MatrixXd::Zero(n_z, n_z);
            for (int i = 0; i < n_z; ++i) {
                Q(i, i) = Gx02.row(i).cwiseAbs().sum(); // 计算每行绝对值和
            }
            
            // 水平拼接矩阵
            MatrixXd G_j(Gx01.rows(), Gx01.cols() + Q.cols());
            G_j << Gx01, Q;
            
            return G_j;
        }
    }

    // 多项式求值辅助函数
    double polyval(const VectorXd& coeffs, double x) {
        double result = 0.0;
        for (int i = 0; i < coeffs.size(); ++i) {
            result = result * x + coeffs[coeffs.size() - 1 - i];
        }
        return result;
    }

    // 多项式导数系数计算
    VectorXd polyder(const VectorXd& coeffs) {
        if (coeffs.size() <= 1) {
            return VectorXd::Zero(1);
        }
        
        VectorXd deriv(coeffs.size() - 1);
        for (int i = 0; i < deriv.size(); ++i) {
            deriv[i] = coeffs[i] * (coeffs.size() - 1 - i);
        }
        return deriv;
    }

    // 离线数据集EKF实现
    void EKF_offline(const MatrixXd& A, const VectorXd& B, double D,
        const VectorXd& Cur, const VectorXd& Vol,
        const VectorXd& pfit, int s,
        const VectorXd& x1_, const MatrixXd& Gw, 
        double Gv, const MatrixXd& Q, 
        double R, MatrixXd P0, 
        const MatrixXd& BG1_diag,
        MatrixXd& x_, MatrixXd& x_l, MatrixXd& x_u,
        VectorXd& y_, VectorXd& y_l, VectorXd& y_u
    ) {
        int N = Cur.size();
        int n_state = x1_.size();
        
        // 计算多项式导数
        VectorXd pder = polyder(pfit);
        
        // 初始化输出变量
        x_ = MatrixXd::Zero(n_state, N);
        x_l = MatrixXd::Zero(n_state, N);
        x_u = MatrixXd::Zero(n_state, N);
        y_ = VectorXd::Zero(N - 1);
        y_l = VectorXd::Zero(N - 1);
        y_u = VectorXd::Zero(N - 1);
        
        // 初始化边界生成器
        MatrixXd BG_current = BG1_diag;
        int r_current = BG1_diag.cols();
        
        // 存储中间结果
        VectorXd Uoc = VectorXd::Zero(N - 1);
        VectorXd Up = VectorXd::Zero(N - 1);
        
        // 设置初始状态
        x_.col(0) = x1_;
        x_l.col(0) = x1_ - BG1_diag.cwiseAbs().rowwise().sum();
        x_u.col(0) = x1_ + BG1_diag.cwiseAbs().rowwise().sum();
        
        // 主循环
        for (int k = 0; k < N - 1; ++k) {
            // 预测步骤
            VectorXd x_pre = A * x_.col(k) + B * Cur[k];
            MatrixXd BG_pre(A * BG_current);
            BG_pre.conservativeResize(NoChange, BG_pre.cols() + Gw.cols());
            BG_pre.rightCols(Gw.cols()) = Gw;
            
            // 计算测量矩阵
            RowVectorXd C(2);
            C << -1, polyval(pder, x_pre[1]);
            
            // 估计输出
            Uoc[k] = polyval(pfit, x_pre[1]);
            Up[k] = x_pre[0];
            y_[k] = Uoc[k] - Up[k] + D * Cur[k];
            
            // 计算输出边界
            MatrixXd BG_y = C * BG_current;
            BG_y.conservativeResize(NoChange, BG_y.cols() + 1);
            BG_y(BG_y.cols() - 1) = Gv;
            BG_y = jiangjie(BG_y, s);
            
            y_l[k] = y_[k] - BG_y.cwiseAbs().sum();
            y_u[k] = y_[k] + BG_y.cwiseAbs().sum();
            
            // 卡尔曼增益和状态更新
            MatrixXd P_pred = A * P0 * A.transpose() + Q;
            double S = (C * P_pred * C.transpose())(0,0) + R; // 标量计算
            MatrixXd K = P_pred * C.transpose() * (1.0/S);        // 用除法替代逆
            P0 = (MatrixXd::Identity(n_state, n_state) - K * C) * P_pred;
            
            // 状态更新
            x_.col(k + 1) = x_pre + K * (Vol[k] - y_[k]);
            
            // 更新边界生成器
            MatrixXd BG(BG_pre);
            BG.conservativeResize(NoChange, BG.cols() + (K * (-BG_y)).cols());
            BG.rightCols((K * (-BG_y)).cols()) = K * (-BG_y);
            BG_current = jiangjie(BG, s);
            
            // 计算状态边界
            VectorXd abs_sum = BG_current.cwiseAbs().rowwise().sum();
            x_l.col(k + 1) = x_.col(k + 1) - abs_sum;
            x_u.col(k + 1) = x_.col(k + 1) + abs_sum;
        }
    }

  
    void save_data_to_csv(
        const Eigen::MatrixXd& x_, const Eigen::MatrixXd& x_l, const Eigen::MatrixXd& x_u,
        const Eigen::VectorXd& Vol, const Eigen::VectorXd& y_, 
        const Eigen::VectorXd& y_l, const Eigen::VectorXd& y_u,
        const Eigen::MatrixXd& x, const Eigen::VectorXd& y, int N)
    {
        std::ofstream file("/home/cat/SOC_FaultDetection_demo/data/output/offline_origin.csv");
        
        file << "k,Up_est,Up_low,Up_up,Up_true,SOC_est,SOC_low,SOC_up,SOC_true,Vol,y_est,y_low,y_up,y_true\n";
        
        for (int i = 0; i < N-1; i++) {
            file << i << ","
                << x_(0, i) << ","
                << x_l(0, i) << ","
                << x_u(0, i) << ","
                << x(0, i) << ","
                << x_(1, i) << ","
                << x_l(1, i) << ","
                << x_u(1, i) << ","
                << x(1, i) << ","
                << Vol(i) << ","
                << y_(i) << ","
                << y_l(i) << ","
                << y_u(i) << ","
                << y(i) << "\n";
        }
        
        file.close();
    }

    int main() {

        VectorXd pfit = load_data("/home/cat/zkf_ws/data/pre_data/pfit_offline.txt");
        VectorXd voltage, current;
        int k = 1; // 计数器，与Python代码一致
        voltage = load_data("/home/cat/zkf_ws/data/pre_data/voltage.txt");
        current = load_data("/home/cat/zkf_ws/data/pre_data/current.txt");
                

        VectorXd Cur = -current;
        int N = Cur.size();
        VectorXd Vol = voltage.array() + 0.0001 * VectorXd::Random(N).array();

        // 参数设置
        double R0 = 0.072, Rp = 0.03, Cp = 1000, Qn = 2 * 3600, deltaT = 1, eta = 1;
        int s = 12, n = 2;
        Matrix2d Gw = Matrix2d::Identity() * 1e-4;
        Gw(1,1) = 1e-4;
        double Gv = 0.001;
        double ww = 0.0001, vv = 0.001;
        Matrix2d Q = Matrix2d::Identity() * 1e-8;
        double R = 0.01;
        Matrix2d P0 = Matrix2d::Identity();
        P0(0,0) = 0.001; P0(1,1) = 1;
        Matrix2d BG1_diag = Matrix2d::Zero();
        BG1_diag.diagonal() << 0.1, 0.001;

        // 初始化状态向量
        Vector2d x1(0.01, 0.8);
        Vector2d x1_(-0.05, 0.7);

        // 计算状态空间矩阵
        Matrix2d A;
        Vector2d B;
        double D;
        calculate_para_matrix(R0, Rp, Cp, Qn, deltaT, eta, A, B, D);

        // 运行状态模拟
        MatrixXd x;
        VectorXd y;
        generate_state(A, B, D, Cur, x1, pfit, ww, vv, x, y);

        // 运行EKF估计
        MatrixXd x_(2, N), x_l(2, N), x_u(2, N);
        VectorXd y_(N-1), y_l(N-1), y_u(N-1);
        EKF_offline(A, B, D, Cur, Vol, pfit, s, 
                x1_, Gw, Gv, Q, R, P0, BG1_diag,
                x_, x_l, x_u, y_, y_l, y_u);

    
        // // ---- 区间判断并即时输出 ----
        // double v_now = Vol[k];                     // 若 Vol[0] 是初始化值，就用 k+1；若没有初始化值，改为 Vol[k]
        // bool out_of_range = (v_now < y_l[k])           // 低于下界？
        //                 || (v_now > y_u[k]);         // 或高于上界？
        // int flag = out_of_range ? 1 : 0;               // 不在区间 → 1，在区间 → 0
        // std::cout << flag << '\n';                     // 实时打印，可替换为写文件／串口等

        
        // 保存
        save_data_to_csv(x_, x_l, x_u, Vol, y_, y_l, y_u, x, y, N);

        return 0;
    }