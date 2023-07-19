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
func solution(_ t: String, _ p: String) -> Int {
    let len = p.count
    var answer = 0
    
    for i in 0..<t.count-len+1 {
        let startIndex = t.index(t.startIndex, offsetBy: i)
        let endIndex = t.index(t.startIndex, offsetBy: i+len-1)
        let range = startIndex...endIndex
        
        if Int64(t[range])! <= Int64(p)! {
            answer += 1
        }
    }
    
    return answer
}
