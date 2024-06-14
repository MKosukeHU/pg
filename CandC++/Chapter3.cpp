// 何度でも入力できて、整数値の符号を表示する
/*
#include <iostream>
#include <string> // 語句を扱う
using namespace std;
int main()
{
    string retry;
    do
    {
        int n;
        cout << "整数値:"; cin >> n;
        if (n > 0)
            cout << "その値は正の数です。\n";
        else if (n < 0)
            cout << "その値は負の数です。\n";
        else
            cout << "その値は０です。\n";
        
        cout << "もう一度？ Y - Yes/N - No:";
        cin >> retry;
    } while (retry == "Y" || retry == "y"); // whileの条件中はdoの処理が繰り返される
} // || は or の意味
*/

// 数当てゲーム
/*
#include <iostream>
#include <ctime>
#include <cstdlib>
using namespace std;
int main()
{
    srand(time(NULL));
    int no = 10 + rand() % 90;
    int x;

    cout << "数当てゲーム開始！\n";
    cout << "10 ~ 99の数を当ててください。\n";

    do{
        cout << "幾つでしょう？:"; cin >> x;
        if (x > no)
            cout << "もっと小さな数です！\n";
        else if (x < no)
            cout << "もっと大きな数です！\n";
    }
    while(x != no);
    cout << "正解です！\n";
}
*/

// ２つの整数値の小さい方から順に大きい方までを順表示する
/*
#include <iostream>
using namespace std;
int main()
{
    int a, b;
    cout << "整数値:"; cin >> a;
    cout << "整数値:"; cin >> b;

    int min, max;
    if (a > b){
        min = b;
        max = a;
    }
    else{
        min = a;
        max = b;
    }
    // if (a > b){int t = a: b = a; b = t}

    do{
        cout << min << " ";
        min = min + 1;
    }
    while(min <= max);
    cout << "\n";
}
*/

// 正の整数値から０までカウントを表示
/*
#include <iostream>
using namespace std;
int main()
{
    int n;
    do
    {
        cout << "正の整数値:"; cin >> n;
        if (n <= 0)
            cout << "入力値が不正です。\n";
    }
    while(n <= 0);

    do{
        cout << n << "\n";
        n = n - 1; // n-- でもいける。n = n + 1はn++でもいける
    }
    while(n != -1);
}
*/

// 整数値個数分だけ*を表示する。１未満のときは何も表示しない
/*
#include <iostream>
using namespace std;
int main()
{
    int n;
    cout << "整数値:"; cin >> n;
    if (n > 0){
        int i = 0;
        while (i < n){
            cout << '*';
            i++;
        }
        cout << "\n";
    }
}
*/

// +-を交互に入力地文だけ表示する
/*
#include <iostream>
using namespace std;
int main()
{
    int n;
    cout << "整数値:"; cin >> n;
    
    if (n > 0){
        int i = 0;
        while (i < n){
            if (i % 2 == 0)
                cout << "+";
            else
                cout << "-";
            i++;
        }
        cout << "\n";
    }
}
*/

// 正の整数値の桁数を表示
/*
#include <iostream>
using namespace std;
int main()
{
    int n;
    cout << "正の整数の桁数を求めます。\n";
    cout << "正の整数:"; cin >> n;
    
    if (n > 0){
        int i = 10;
        int digits = 1;
        while (n >= i){
            i = i * 10;
            digits++;
        }
        cout << "入力された数値の桁数は" << digits << "です。\n";
    }
}
*/

// 入力値個数分だけ*を表示する、for文で
#include <iostream>
using namespace std;
int main()
{
    int n;
    cout << "何個＊を表示しますか？:"; cin >> n;
    
    if (n > 0){
        for (int i = 0; i < n; i++){
            cout << "*";
        }
        cout << "\n";
    }
}