// 読み込んだ整数値の絶対値を求める
/*
#include <iostream>
using namespace std;
int main()
{
    int n;
    cout << "整数値:"; cin >> n;
    int abs = n;
    if (n < 0)
        abs = -n;
    cout << "その整数値の絶対値は" << abs << "です。\n";
}
*/

// 読み込んだ整数値の符号を判定する
/*
#include <iostream>
using namespace std;
int main()
{
    int n;
    cout << "整数値:"; cin >> n;
    if (n == 0)
        cout << "符号は0です。\n";
    else if (n > 0)
        cout << "符号は正です。\n";
    else
        cout << "符号は負です。\n";        
}
*/

// 入力さる整数値の大小関係を表示する
/*
#include <iostream>
using namespace std;
int main()
{
    int a;
    int b;
    cout << "a ="; cin >> a;
    cout << "b ="; cin >> b;

    if (a > b)
        cout << a << "の方が大きいです。\n";
    else if (b > a)
        cout << b << "の方が大きいです。\n";
    else
        cout << a << "と" << b << "は同じ大きさです。\n";
}
*/

// 5の倍数か判定する
/*
#include <iostream>
using namespace std;
int main()
{
    int n;
    cout << "整数値:"; cin >> n;
    
    if (n <= 0)
        cout << "正でない整数値が入力されました。\n";
    else if (n % 5 == 0)
        cout << n << "は５の倍数です。\n";
    else
        cout << n << "は５の倍数ではありません。\n";
}
*/

// 3で割った剰余を表示する
/*
#include <iostream>
using namespace std;
int main()
{
    int n;
    cout << "整数値:"; cin >> n;
    if (n <= 0)
        cout << "正でない整数値が入力されました。\n";
    else if (n % 3 == 0)
        cout << n << "は３の倍数です。\n";
    else
        cout << n << "を3で割った余は" << n % 3 << "です。\n";
}
*/

// 入力値を評価
/*
#include <iostream>
using namespace std;
int main()
{
    int n;
    cout << "成績値:"; cin >> n;
    if (0 <= n && n < 60)
        cout << "不可\n";
    else if (60 <= n && n < 70)
        cout << "可\n";
    else if (70 <= n && n < 80)
        cout << "良\n";
    else if (80 <= n && n < 90)
        cout << "優\n";
    else if (90 <= n)
        cout << "秀\n";
    else
        cout << "不正な点数が入力されました。\n";
}
*/

// 実数値の大きい方を表示する
/*
#include <iostream>
using namespace std;
int main()
{
    double a, b;
    cout << "実数値:"; cin >> a;
    cout << "実数値:"; cin >> b;
    
    if (a > b)
        cout << "大きい方の値は" << a << "です。\n";
    else
        cout << "大きい方の値は" << b << "です。\n";
    
    
    double max = a > b ? a : b; // pythonで言うところの max = a if a > b else b
    cout << "大きい方の値は" << max << "です。\n";
}
*/

// ２つの整数値の差を表示する
// 差が10以上かどうか判定する
/*
#include <iostream>
using namespace std;
int main()
{
    int a, b;
    cout << "a = "; cin >> a;
    cout << "b = "; cin >> b;
    int diff = a > b ? a - b : b - a;

    // cout << a << "と" << b << "の差は" << diff << "です。\n";

    if (diff > 10)
        cout << a << "と" << b << "の差は11以上です。\n";
    else
        cout << a << "と" << b << "の差は10以下です。\n";
}
*/

// 3つの整数値のうちの最小値を表示する
/*
#include <iostream>
using namespace std;
int main()
{
    int a, b, c;
    cout << "a = "; cin >> a;
    cout << "b = "; cin >> b;
    cout << "c = "; cin >> c;

    int min = a;
    if (min > b)
        min = b;
    if (min > c)
        min = c;
    cout << "最小値は" << min << "です。\n";
}
*/

// 3つの整数値の中央値を表示する
// 先に条件分岐の樹形図を書くとわかりやすい
/*
#include <iostream>
using namespace std;
int main()
{
    int a, b, c;
    cout << "整数値:"; cin >> a;
    cout << "整数値:"; cin >> b;
    cout << "整数値:"; cin >> c;

    int med;
    if (a >= b)
        if (b >= c)
            med = b;
        else if (a <= c)
            med = a;
        else
            med = c;
    else if (a > c)
        med = a;
    else if (b > c)
        med = c;
    else 
        med = b;
    cout << "中央値は" << med << "です。\n";
}
*/

// ２つの整数値の大きい方と小さい方を明示する
/*
#include <iostream>
using namespace std;
int main()
{
    int a, b;
    cout << "整数値:"; cin >> a;
    cout << "整数値:"; cin >> b;

    int max, min;

    if (a > b){
        max = a;
        min = b;
    }
    else {
        max = b;
        min = a;
    }
    // 条件文内での処理が２行以上のときは{}で処理を囲う。

    if (a == b)
        cout << "二つの値は同じです。\n";
    else
        cout << "小さい方の値は" << min << "で、大きい方の値は" << max << "です。\n";
}
*/

// 整数値が３の倍数か判定、剰余も表示
/*
#include <iostream>
using namespace std;
int main()
{
    int n;
    cout << "整数値:"; cin >> n;
    
    if (n % 3 == 0)
        cout << n << "は３で割り切れます。\n";
    else
        cout << n << "は３で割り切れません。剰余は" << n % 3 << "です。\n";
}
*/

// 実数値が2.5で割り切れるか判定、剰余も表示
/*
#include <iostream>
#include <cmath>
using namespace std;
int main()
{
    double n;
    cout << "実数値:"; cin >> n;

    if (double m = fmod(n, 2.5)){
        cout << n << "は2.5で割り切れません。剰余は" << m << "です。\n";
    }
    else 
        cout << n << "は2.5で割り切れます。\n";
}
*/

// 0, 1, 2をぐー、ちょき、パーに対応させて表示する
/*
#include <iostream>
#include <ctime>
#include <cstdlib>
using namespace std;
int main()
{
    srand(time(NULL)); // srand is random seed
    int hand = rand() % 3; // 剰余は0, 1, 2のいずれか
    
    switch (hand) { // swith文ではcaseとして条件を表示できる
        case 0: cout << "グー\n"; break; // case 0:の意味は、if hand == 0
        case 1: cout << "チョキ\n"; break; // breakで処理を終了する
        case 2: cout << "パー\n"; break;
    }
}
*/

// 入力月の季節を判定
#include <iostream>
using namespace std;
int main()
{
    int month;
    cout << "何月ですか？:"; cin >> month;

    switch (month)
    {
        case 3: case 4: case 5: cout << "春です。\n"; break;
        case 6: case 7: case 8: cout << "夏です。\n"; break;
        case 9: case 10: case 11: cout << "秋です。\n"; break;
        case 12: case 1: case 2: cout << "冬です。\n"; break;
        default : cout << "そんな月はありませんよ！\n"; break; // どの条件とも一致しないときの処理
    }
}