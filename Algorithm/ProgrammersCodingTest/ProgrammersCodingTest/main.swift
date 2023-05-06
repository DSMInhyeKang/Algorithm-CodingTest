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
func solution(_ today:String, _ terms:[String], _ privacies:[String]) -> [Int] {
    var answer = [Int]()

    let today = getIntValueFromDateString(date: today)
    let expiration = terms.reduce(into: [String: Int]()) { dict, str in
        let info = str.components(separatedBy: " ")
        dict[info[0]] = Int(info[1])!
    }

    privacies.enumerated().forEach { (offset, privacy) in
        let temp = privacy.components(separatedBy: " ")
        let beginDate = getIntValueFromDateString(date: temp[0])
        let isExpired =  beginDate + expiration[temp[1]]! * 28 <= today
        if isExpired { answer.append(offset + 1) }
    }

    return answer
}


func getIntValueFromDateString(date: String) -> Int {
    let info = date.components(separatedBy: ".").map { Int($0)! }
    return info[0] * 12 * 28 + info[1] * 28 + info[2]
}
