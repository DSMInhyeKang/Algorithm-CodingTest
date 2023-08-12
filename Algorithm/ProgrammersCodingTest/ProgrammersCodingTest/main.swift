//
//  main.swift
//  ProgrammersCodingTest
//
//  Created by 강인혜 on 2023/04/30.
//

import Foundation

// 중복된 문자 제거(120888)
//func solution(_ my_string: String) -> String {
//    var result = ""
//    for str in my_string where !result.contains(str) {
//        result.append(str)
//    }
//    return result
//}



// 개인정보 수집 유효기간(150370)
//func solution(_ today:String, _ terms:[String], _ privacies:[String]) -> [Int] {
//    var answer = [Int]()
//
//    let today = getIntValueFromDateString(date: today)
//    let expiration = terms.reduce(into: [String: Int]()) { dict, str in
//        let info = str.components(separatedBy: " ")
//        dict[info[0]] = Int(info[1])!
//    }
//
//    privacies.enumerated().forEach { (offset, privacy) in
//        let temp = privacy.components(separatedBy: " ")
//        let beginDate = getIntValueFromDateString(date: temp[0])
//        let isExpired =  beginDate + expiration[temp[1]]! * 28 <= today
//        if isExpired { answer.append(offset + 1) }
//    }
//
//    return answer
//}
//
//
//func getIntValueFromDateString(date: String) -> Int {
//    let info = date.components(separatedBy: ".").map { Int($0)! }
//    return info[0] * 12 * 28 + info[1] * 28 + info[2]
//}


//신고 결과 받기(92334)
//func solution(_ id_list: [String], _ report: [String], _ k: Int) -> [Int] {
//    var reported: [String: Int] = [:]
//    var user: [String: [String]] = [:]
//
//    for rep in Set(report) { //Set으로 중복 제거하면 시간 초과 안 뜸
//        let splited = rep.split(separator: " ").map { String($0) }
//        user[splited[0]] = (user[splited[0]] ?? []) + [splited[1]] //prosecutor
//        reported[splited[1]] = (reported[splited[1]] ?? 0) + 1 //suspect
//    }
//
//    return id_list.map { id in
//        return (user[id] ?? []).reduce(0) {
//            $0 + ((reported[$1] ?? 0) >= k ? 1 : 0)
//        }
//    }
//}


// 공원 산책(172928)
//func solution(_ park: [String], _ routes: [String]) -> [Int] {
//    var (park, x, y) = (park.map { $0.map { String($0) } }, 0, 0)
//
//    for i in 0..<park.count {
//        for j in 0..<park[0].count where park[i][j] == "S" {
//             (x, y) = (i, j)
//        }
//    }
//
//    func isIn(_ x: Int, _ y: Int) -> Bool {
//        x >= 0 && y >= 0 && x < park.count && y < park[0].count
//    }
//
//    for d in routes {
//        let d = d.split { $0 == " "}.map { String ($0) }
//        let (dir, wei) = (d[0], d[1])
//        var (ixiy, skip) = ((0, 0), false)
//
//        switch dir {
//            case "N":
//                ixiy = (-1, 0)
//            case "S":
//                ixiy = (1, 0)
//            case "W":
//                ixiy = (0, -1)
//            default:
//                ixiy = (0, 1)
//        }
//
//        var (X, Y) = (x, y)
//
//        for _ in 0..<Int(wei)! where !skip {
//            (X, Y) = (X + ixiy.0, Y + ixiy.1)
//            if !isIn(X, Y) { skip = true; break }
//            if park[X][Y] == "X" { skip = true }
//        }
//
//        if skip { continue }
//        (x, y) = (X, Y)
//    }
//
//    return [x, y]
//}


// 수박수박수박수박수박수?(12922)
//func solution(_ n:Int) -> String {
//    return (0..<n).map{($0%2==0 ? "수":"박")}.reduce("", +)
//}



// 신규 아이디 추천(72410)
//func solution(_ new_id:String) -> String {
//    var new = ""
//
//    new_id.forEach {
//        if $0.isLowercase || Int(String($0)) != nil || $0 == "." || $0 == "_" || $0 == "-" {
//            if !(new.isEmpty && $0 == ".") && !(new.last == "." && $0 == ".") {
//                new.append($0)
//            }
//        } else {
//            if $0.isUppercase {
//                new.append($0.lowercased())
//            }
//        }
//    }
//
//    if new.last == "." {
//        new.removeLast()
//    }
//
//    if new.count > 15 {
//        new.removeSubrange(new.index(after: new.index(new.startIndex, offsetBy: 14)) ..< new.endIndex)
//    } else if new.count <= 2{
//        if new.count <= 2 {
//            if new.isEmpty {
//                new = "a"
//            }
//
//            while new.count < 3 {
//                new.append(new.last!)
//            }
//        }
//    }
//
//
//    if new.last == "." {
//        new.removeLast()
//    }
//
//    return new
//}



// 달리기 경주(178871)
//func solution(_ players:[String], _ callings:[String]) -> [String] {
//    var players = players
//    var callings = callings
//    var dict = [String:Int]()
//
//    for i in 0..<players.count {
//        dict[players[i]] = i
//    }
//
//    for j in 0..<callings.count {
//        var rank = dict[callings[j]]!
//        let name  = players[rank - 1]
//        players[rank - 1] = callings[j]
//        players[rank] = name
//        dict[callings[j]]! -= 1
//        dict[players[rank]]! += 1
//    }
//
//    return players
//}



// 추억 점수(176963)
//func solution(_ name:[String], _ yearning:[Int], _ photo:[[String]]) -> [Int] {
//    let score: [String: Int] = Dictionary(uniqueKeysWithValues: zip(name, yearning))
//
//    return photo.map { $0.reduce(0) { $0 + (score[$1] ?? 0) } }
//}




// 카드 뭉치(159994)
//func solution(_ cards1:[String], _ cards2:[String], _ goal:[String]) -> String {
//    var index1:Int = 0
//    var index2:Int = 0
//
//    for i in 0..<goal.count {
//        if index1 < cards1.count && goal[i] == cards1[index1] {
//            index1 += 1
//        } else if index2 < cards2.count && goal[i] == cards2[index2] {
//            index2 += 1
//        } else {
//            return "No"
//        }
//    }
//
//    return "Yes"
//}



// 크기가 작은 부분 문자열(147355)
//func solution(_ t: String, _ p: String) -> Int {
//    let len = p.count
//    var answer = 0
//
//    for i in 0..<t.count-len+1 {
//        let startIndex = t.index(t.startIndex, offsetBy: i)
//        let endIndex = t.index(t.startIndex, offsetBy: i+len-1)
//        let range = startIndex...endIndex
//
//        if Int64(t[range])! <= Int64(p)! {
//            answer += 1
//        }
//    }
//
//    return answer
//}



// 대소문자 바꿔서 출력하기(181949)
//print(readLine()!.map { $0.isUppercase ? $0.lowercased() : $0.uppercased() }.joined())



// 덧칠하기(161989)
//func solution(_ n: Int, _ m: Int, _ section: [Int]) -> Int {
//    var section = section
//    var count = 0
//
//    while !section.isEmpty {
//        let wall = section[0]
//
//        for _ in 0..<m {
//            guard let first = section.first else { break }
//
//            if first < wall + m {
//                let _ = section.removeFirst()
//            } else {
//                break
//            }
//        }
//
//        count += 1
//    }
//    return count
//}



// 삼총사(131705)
//func solution(_ number: [Int]) -> Int {
//    var answer = 0
//
//    for i in 0..<number.count {
//        for j in i+1..<number.count {
//            for k in j+1..<number.count {
//                if number[i] + number[j] + number[k] == 0 { answer += 1 }
//            }
//        }
//    }
//
//    return answer
//}



// 성격 유형 검사하기(118666)
//func solution(_ survey: [String], _ choices: [Int]) -> String {
//    let types = ["R", "T", "C", "F", "J", "M", "A", "N"]
//    var score = Array(repeating: 0, count: types.count)
//    var result = ""
//
//    for i in 0..<survey.count {
//        if choices[i] == 4 {
//            continue
//        } else if choices[i] < 4 {
//            score[types.firstIndex(of: String(survey[i].first!))!] += (4 - choices[i])
//        } else {
//            score[types.firstIndex(of: String(survey[i].last!))!] += (choices[i] - 4)
//        }
//    }
//
//    for i in stride(from: 0, to: score.count, by: 2) {
//        if score[i] >= score[i+1] { // 삼항연산자 -> 시간 초과
//            result += types[i]
//        } else {
//            result += types[i+1]
//        }
//    }
//
//    return result
//}



// 실패율(42889)
//func solution(_ N: Int, _ stages: [Int]) -> [Int] {
//    var failure = [Int: Double]()
//    var total = Double(stages.count)
//    var countArr = Array(repeating: 0, count: N+2)
//
//    for num in stages {
//        countArr[num] += 1
//    }
//
//    for i in 1..<N+1 { //filter map 고차함수 쓰면 시간 초과 -> for문 안에 최대한 사용 자제
//        if countArr[i] == 0 {
//            failure[i] = 0.0
//        } else {
//            total = total - Double(countArr[i])
//            failure[i] = Double(countArr[i]) / total
//        }
//    }
//
//    let sortArr = failure.sorted(by: <).sorted(by: { $0.1 > $1.1 })
//    let result = sortArr.map{ $0.key }
//
//    return result
//}



// 크레인 인형뽑기 게임(64061)
//func solution(_ board: [[Int]], _ moves: [Int]) -> Int {
//    var board = board
//    var basket = [Int]()
//    var result = 0
//
//    for move in moves {
//        var y = 0
//
//        while y < board.count {
//            let doll = board[y][move-1]
//
//            if doll == 0 {
//                y += 1
//                continue
//            }
//            basket.append(doll)
//            board[y][move-1] = 0
//
//            if board[y][move-1] == 0 { break }
//        }
//
//        if (basket.count >= 2) && (basket[basket.endIndex-1] == basket[basket.endIndex-2]) {
//            basket.removeLast(2) // popLast: 옵셔널 리턴
//            result += 2
//        }
//    }
//
//    return result
//}



// 기사단원의 무기(136798)
//func solution(_ number: Int, _ limit: Int, _ power: Int) -> Int {
//    var result = [Int]()
//
//    for number in 1...number{
//        var count = 0
//
//        for i in 1...Int(sqrt(Double(number))) {
//            if number % i == 0 {
//                if(i * i) == number {
//                    count += 1
//                } else {
//                    count += 2
//                }
//            }
//        }
//
//        count = count > limit ? power : count
//        result.append(count)
//    }
//
//    return result.reduce(0){ $0 + $1 }
//}



// 콜라 문제(132267)
//func cokes(_ a: Int, _ b: Int, _ n: Int) -> Int {
//    var rest = n%a
//    var count = (n/a)*b
//
//    if a > n { return 0 }
//
//    let bottles = rest + count
//
//    return count + cokes(a, b, bottles)
//}
//
//func solution(_ a: Int, _ b: Int, _ n: Int) -> Int {
//    return cokes(a, b, n)
//}



// 과일 장수(135808)
//func solution(_ k: Int, _ m: Int, _ score: [Int]) -> Int {
//    let s = score.sorted(by: >)
//
//    return stride(from: m-1, to: score.count, by: m).reduce(0) { $0 + s[$1] * m }
//}



// 문자열 나누기(140108)
//func solution(_ s: String) -> Int {
//    var answer = 0
//    var x: Character? = nil
//    var xCount = 0
//
//    for i in s {
//        if x == nil {
//            x = i
//            xCount = 1
//            answer += 1
//            continue
//        }
//
//        xCount += x == i ? 1 : -1
//
//        if xCount == 0 {
//            x = nil
//        }
//    }
//
//    return answer
//}



// 나머지가 1이 되는 수 찾기(87389)
//func solution(_ n: Int) -> Int {
//    for num in 2...n  {
//        if n % num == 1 {
//            return num
//        }
//    }
//
//    return 1
//}



// 문자열 곱하기(181940)
//func solution(_ my_string: String, _ k: Int) -> String {
//    return String(repeating: my_string, count: k)
//}



// 홀짝에 따라 다른 값 반환하기(181935)
//func solution(_ n: Int) -> Int {
//    return n % 2 == 0 ? stride(from: 0, through: n, by: 2).map { $0 * $0 }.reduce(0, +) : stride(from: 1, through: n, by: 2).reduce(0, +)
//}



// 공배수(181936)
//func solution(_ number: Int, _ n: Int, _ m: Int) -> Int {
//    return number % n == 0 && number % m == 0 ? 1 : 0
//}



// 문자 리스트를 문자열로 반환하기(181941)
//func solution(_ arr: [String]) -> String {
//    return arr.joined()
//}



// 덧셈식 출력하기(181947)
//let n = readLine()!.components(separatedBy: [" "]).map { Int($0)! }
//let (a, b) = (n[0], n[1])
//
//print(a, "+", b, "=", a+b)


// 문자열 섞기(181942)
//func solution(_ str1:String, _ str2:String) -> String {
//    return zip(str1, str2).map { String($0) + String($1) }.joined()
//}



// 홀짝 구분하기(181944)
//let n = Int(readLine()!)!
//
//print(n, "is", n%2 == 0 ? "even" : "odd")



// n의 배수(181937)
//func solution(_ num: Int, _ n: Int) -> Int {
//    return num % n == 0 ? 1 : 0
//}



// flag에 따라 다른 값 반환하기(181933)
//func solution(_ a: Int, _ b: Int, _ flag: Bool) -> Int {
//    return flag ? a+b : a-b
//}



// 문자열 붙여서 출력하기(181946)
//let inp = readLine()!.components(separatedBy: [" "]).map { $0 }
//
//print(inp.joined())



// 조건 문자열(181934)
//func solution(_ ineq: String, _ eq: String, _ n: Int, _ m: Int) -> Int {
//    switch ineq+eq {
//    case ">=":
//        return n >= m ? 1 : 0
//    case ">!":
//        return n > m ? 1 : 0
//    case "<=":
//        return n <= m ? 1 : 0
//    case "<!":
//        return n < m ? 1 : 0
//    default:
//        return -1
//    }
//}



// 더 크게 합치기(181939)
//func solution(_ a: Int, _ b: Int) -> Int {
//    return max(Int(String(a) + String(b))!, Int(String(b) + String(a))!)
//}



// 두 수의 연산값 비교하기(181938)
//func solution(_ a: Int, _ b: Int) -> Int {
//    return max(Int(String(a) + String(b))!, 2 * a * b)
//}



// 문자열 겹쳐쓰기(181943)
//func solution(_ my_string: String, _ overwrite_string: String, _ s: Int) -> String {
//    var myString = Array(my_string)
//    myString.replaceSubrange(s...(overwrite_string.count + s - 1), with: Array(overwrite_string))
//
//    return String(myString)
////    return String(my_string.prefix(s)) + overwrite_string + String(my_string.suffix(my_string.count - overwrite_string.count - s))
//}



// 서울에서 김서방 찾기(12919)
//func solution(_ seoul: [String]) -> String {
//    return "김서방은 \(seoul.firstIndex(of: "Kim")!)에 있다"
//}



// 자릿수 더하기(12931)
//func solution(_ n: Int) -> Int {
//    var answer: Int = 0
//
//    for i in String(n) {
//        answer += Int(String(i))!
//    }
//
//    return answer
//
////    return String(n).reduce(0, { $0 + Int(String($1))! });
//}



// 정수 내림차순으로 배치하기(12933)
//func solution(_ n: Int64) -> Int64 {
//    return Int64(String(String(n).sorted(by: >)))!
//}



// 시저 암호(12926)
//func solution(_ s: String, _ n: Int) -> String {
//    let alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".map { $0 }
//
//    return String(
//        s.map {
//            guard let index = alphabet.firstIndex(of: Character($0.uppercased())) else { return $0 }
//
//            let word = alphabet[(index + n) % alphabet.count]
//
//            return $0.isLowercase ? Character(word.lowercased()) : word
//        }
//    )
//}



// 문자열 내 마음대로 정렬하기(12915)
//func solution(_ strings: [String], _ n: Int) -> [String] {
//    return strings.sorted {
//        Array($0)[n] == Array($1)[n] ? $0 < $1 : Array($0)[n] < Array($1)[n]
//    }
//}



// 올바른 괄호(12909)
//func solution(_ s: String) -> Bool {
//    var opens = 0
//
//    for c in s {
//        if c == "(" {
//            opens += 1;
//        } else if c == ")" {
//            opens -= 1;
//
//            if opens < 0 {
//                return false;
//            }
//        }
//    }
//    return opens == 0;
//}



// 가장 큰 정사각형 찾기(12905)
//func solution(_ board:[[Int]]) -> Int {
//    var answer:Int = 0
//    var dp = board
//
//    for i in 0..<board.count {
//        for j in 0..<board[0].count {
//            answer = max(answer, board[i][j])
//            if i - 1 < 0 || j - 1 < 0 {
//                continue
//            }
//
//            if board[i][j] == 1 && board[i-1][j] == 1 && board[i][j-1] == 1 {
//                dp[i][j] = min(dp[i][j-1], dp[i-1][j], dp[i-1][j-1]) + 1
//                answer = max(answer, dp[i][j])
//            }
//        }
//    }
//
//    return answer * answer
//}



// 땅따먹기(12913)
//func solution(_ land: [[Int]]) -> Int {
//    var dp = land
//
//    for i in 1..<land.count {
//        for j in 0..<land[0].count {
//            for k in 0..<land[0].count {
//                if j == k { continue }
//                dp[i][j] = max(dp[i][j], dp[i-1][k] + land[i][j])
//            }
//        }
//    }
//
//    return dp.last!.max()!
//}



// 가운데 글자 가져오기(12903)
//func solution(_ s: String) -> String {
//    if s.count % 2 == 0 {
//        return String(Array(s)[(s.count/2 - 1)...(s.count/2)])
//    } else {
//        return String(Array(s)[s.count/2])
//    }
////    return String(s[String.Index(encodedOffset: (s.count-1)/2)...String.Index(encodedOffset: s.count/2)])
//}



// 다음 큰 숫자(12911)
//func solution(_ n: Int) -> Int {
//    var answer : Int = n + 1
//
//    while true {
//        if n.nonzeroBitCount == answer.nonzeroBitCount {
//            break;
//        }
//        answer += 1
//    }
//
//    return answer
//}



// 내적(70128)
func solution(_ a: [Int], _ b: [Int]) -> Int {
    return zip(a, b).map(*).reduce(0, +)
}
