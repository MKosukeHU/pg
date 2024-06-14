// 画面への出力を行うプログラム
/*
#include <iostream>
using namespace std;
int main()
{
    cout << "初めてのC++プログラム。\n";
    cout << "画面に出力しています。\n";
}
*/

// ファイルのコンパイルを行う際の方法
// 0 ファイルの変更内容を保存する。
// 1 実行したいファイルのあるディレクトリに移動する。
// 2 ターミナルで 「g++ -o <ファイル名> <ファイル名>.cpp」 を実行。
// 3 ターミナルで 「./<ファイル名>」 を実行する。

// 以下、ターミナルでの実行例；
// miyoshikosuke@miyoshiakiratasukunoMacBook-Pro CandC++ % g++ -o helloworld helloworld.cpp
//
// miyoshikosuke@miyoshiakiratasukunoMacBook-Pro CandC++ % ./helloworld
// 初めてのC＋＋プログラム。
// 画面に出力しています。
// miyoshikosuke@miyoshiakiratasukunoMacBook-Pro CandC++ % 

// 氏名を縦向きに表示する
/*
#include <iostream>
using namespace std;
int main()
{
    cout << "三\n好\n\n晃\n輔\n";
}
*/

// ２つの数のx, yの合計と平均を表示する。
/*
#include <iostream>
using namespace std;
int main()
{
    
    //int x; // 数字は整数型（int型）
    //int y;
    //int z;
    
    int x = 63; // 値がわかっているなら最初から初期化して良い
    int y = 18;
    int z = 34;
    cout << "xの値は" << x << "です。\n";
    cout << "yの値は" << y << "です。\n";
    cout << "zの値は" << z << "です。\n";
    cout << "合計は" << x + y + z << "です。\n";
    cout << "平均は" << (x + y + z)/3 << "です。\n";
}
*/

// 数値をキーボード入力にする
/*
#include <iostream>
using namespace std;
int main()
{
    int x; // 読み込む値の型を宣言
    cout << "整数値："; // 数値の入力を促す
    cin >> x; // xに整数値を読み込む
    cout << x << "と入力しましたね。\n"; // xの値を反復して表示
    // cout << x << "に10を加えた値は" << x + 10 << "です。\n";
    // cout << x << "から10を減じた値は"  << x - 10 << "です。\n";
    cout << x << "の最下位桁は" << x % 10 << "です。\n";
}
*/

// 実数値を入力してその平均と合計を計算
/*
#include <iostream>
using namespace std;
int main()
{
    double x; // doubleは実数値型
    double y;
    // cout << "xの値:";
    // cin >> x;
    // cout << "yの値:";
    // cin >> y;
    cout << "xとyの値:";
    cin >> x >> y; // xを入力したらEnterキーを押してyの入力に進む仕様
    cout << "xとyの合計値は" << x + y << "です。\n"; // 計算結果の有効数字は桁数の大きい方に合わせられる
    cout << "xとyの平均値は" << (x + y)/2 << "です。\n";
}
*/

// 三角形の面積を計算する
/*
#include <iostream>
using namespace std;
int main()
{
    double width;
    double height;
    cout << "底辺:"; cin >> width;
    cout << "高さ:"; cin >> height;
    double erea = width * height / 2.0;
    cout << "三角形の面積は" << erea << "です。\n";
}
*/

// 球の表面積と体積を計算する
/*
#include <iostream>
using namespace std;
int main()
{
    const double pi = 3.14; // const宣言を行ったpiには代入などの行為が無効になる
    double r;
    cout << "球の半径:"; cin >> r;
    cout << "球の表面積は" << 4 * pi * r * r << "です。\n";
    cout << "球の体積は" << 4 * pi * r * r * r / 3 << "です。\n";
}
*/

// 1桁あるいは2桁の乱数を生成
/*
#include <ctime>
#include <cstdlib>
#include <iostream>
using namespace std;
int main()
{
    srand(time(NULL)); // 乱数の種を生成
    int n;
    cout << "整数値:"; cin >> n;
    cout << "入力された整数値" << n << "の±5の範囲の乱数を生成しました。それは" << n - 5 + rand() % 11 << "です。\n";
    // cout << "1桁の正の乱数は" << 1 + rand() % 9 << "です。\n"; // rand()は0以上のランダムな整数値を生成する
    // cout << "1桁の負の乱数は" << -1 - rand() % 9 << "です。\n";
    // cout << "2桁の正の乱数は" << 10 + rand() % 90 << "です。\n";
}
*/

// 名前のイニシャルを読み込んで挨拶する
/*
#include <iostream>
using namespace std;
int main()
{
    char ini1, ini2; // 文字を読み込むchar型。characterの略。
    cout << "姓のイニシャルは"; cin >> ini1;
    cout << "名のイニシャルは"; cin >> ini2;
    cout << "\aこんにちは" << ini2 << "." << ini1 << "さん。\n"; // \aは警報音を鳴らすコマンド
}
*/

// 名前を読み込んで挨拶する
#include <string> // 文字列を扱う型
#include <iostream>
using namespace std;
int main()
{
    string last, first;
    cout << "姓は"; cin >> last;
    cout << "名は"; cin >> first;
    cout << "こんにちは" << last << first << "さん。\n";
}



