"use client";

import { useEffect, useState } from "react";
import axios from "axios";
import Cookies from "js-cookie";
import { useRouter } from "next/navigation";

const SERVER_URL = process.env.NEXT_PUBLIC_API_URL;

export default function AuthPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [username, setUsername] = useState("");
  const [isLogin, setIsLogin] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const router = useRouter();

  useEffect(() => {
    const uid = Cookies.get("uid");
    if (uid) {
      router.push("/dashboard");
    }
  }
  , []);



  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError("");

    if (!email || !password || (!isLogin && !username)) {
      setError("Please fill in all required fields");
      setIsLoading(false);
      return;
    }

    try {
      if (isLogin) {
        const response = await axios.post(`${SERVER_URL}/login`, {
          email,
          password,
        });

        if (response.status === 200) {
          const { uid } = response.data;
          Cookies.set("uid", uid, { expires: 7 });
          router.push("/dashboard");
          window.location.reload();
        } else {
          setError(response.data.message || "Login failed");
        }
      } else {
        const response = await axios.post(`${SERVER_URL}/create_user`, {
          email,
          password,
          name: username,
        });

        if (response.status === 201) {
          const { uid } = response.data;
          Cookies.set("uid", uid, { expires: 7 });
          router.push("/dashboard");
          window.location.reload();
        } else {
          setError(response.data.message || "Signup failed");
        }
      }
    } catch (err) {
      setError(err.response?.data?.message || "Something went wrong");
    } finally {
      setIsLoading(false);
    }
 
 
 
 
 
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-50 px-4">
      <div className="w-full max-w-md bg-white p-8 rounded-lg shadow-md">
        <h2 className="text-2xl font-bold text-start mb-6">
          {isLogin ? "Login" : "Sign Up"}
        </h2>

        <form className="space-y-5" onSubmit={handleSubmit}>
          {!isLogin && (
            <div>
              <label htmlFor="username" className="block text-sm font-medium text-gray-700">
                Full Name
              </label>
              <input
                type="text"
                id="username"
                className="mt-1 block w-full px-4 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-black-500 focus:border-black-500"
                placeholder="John Doe"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
              />
            </div>
          )}

          <div>
            <label htmlFor="email" className="block text-sm font-medium text-gray-700">
              Email
            </label>
            <input
              type="email"
              id="email"
              className="mt-1 block w-full px-4 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-black-500 focus:border-black-500"
              placeholder="you@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </div>

          <div>
            <div className="flex justify-between items-center">
              <label htmlFor="password" className="block text-sm font-medium text-gray-700">
                Password
              </label>
             
            </div>
            <input
              type="password"
              id="password"
              className="mt-1 block w-full px-4 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-black-500 focus:border-black-500"
              placeholder="••••••••"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </div>

          {error && <p className="text-sm text-red-600">{error}</p>}

          <button
            type="submit"
            className="w-full bg-black text-white py-2 px-4 rounded-md hover:bg-muted/50 hover:text-black hover:border font-semibold transition"
            disabled={isLoading}
          >
            {isLoading ? (isLogin ? "Logging in..." : "Signing up...") : isLogin ? "Login" : "Sign Up"}
          </button>

       
        </form>

        <p className="text-sm text-center text-gray-600 mt-6">
          {isLogin ? (
            <>Don’t have an account? <button onClick={() => setIsLogin(false)} className="text-black hover:underline">Sign up</button></>
          ) : (
            <>Already have an account? <button onClick={() => setIsLogin(true)} className="text-black hover:underline">Login</button></>
          )}
        </p>
      </div>
    </div>
  );
}
